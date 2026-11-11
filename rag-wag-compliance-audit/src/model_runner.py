"""
Model runner for the five comparison systems:
1. llm_only
2. static_rag
3. wag_only
4. hybrid_rag_wag
5. hybrid_rag_wag_verifier
"""

from __future__ import annotations

from typing import Dict, Any, List
from pathlib import Path
import json

from .common import safe_json_loads
from .retrievers import LexicalRetriever, evidence_to_text
from .query_classifier import classify_query
from .evidence_ranker import rank_evidence
from .conflict_detector import detect_conflicts, check_effective_date
from .answer_generator import generate_answer
from .llm_client import call_llm
from .runtime_verifier import apply_runtime_verifier


class ModelRunner:
    def __init__(
        self,
        retriever: LexicalRetriever,
        prompts_dir: str | Path,
        top_k: int = 5,
        use_mock: bool = True,
        enable_wag: bool = True,
        enable_rag: bool = True,
        enable_verifier: bool = True,
        enable_date_check: bool = True,
        enable_source_ranking: bool = True,
        model_name: str = "gpt-5.4-mini",
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ):
        self.retriever = retriever
        self.prompts_dir = Path(prompts_dir)
        self.top_k = top_k
        self.use_mock = use_mock
        self.enable_wag = enable_wag
        self.enable_rag = enable_rag
        self.enable_verifier = enable_verifier
        self.enable_date_check = enable_date_check
        self.enable_source_ranking = enable_source_ranking
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _retrieve_rag(self, question: str) -> List[Dict[str, Any]]:
        return self.retriever.retrieve(question, top_k=self.top_k, source_filter="internal")

    def _retrieve_wag(self, question: str) -> List[Dict[str, Any]]:
        return self.retriever.retrieve(question, top_k=self.top_k, source_filter="external")

    def _prompt_name(self, model_variant: str) -> str:
        if model_variant == "llm_only":
            return "llm_only_prompt.txt"
        if model_variant == "static_rag":
            return "static_rag_prompt.txt"
        if model_variant == "wag_only":
            return "wag_only_prompt.txt"
        return "hybrid_rag_wag_prompt.txt"

    def _build_prompt(self, model_variant: str, question: str, rag_evidence: List[Dict[str, Any]], wag_evidence: List[Dict[str, Any]]) -> str:
        path = self.prompts_dir / self._prompt_name(model_variant)
        template = path.read_text(encoding="utf-8")
        return (
            template
            .replace("{question}", question)
            .replace("{rag_evidence}", evidence_to_text(rag_evidence))
            .replace("{wag_evidence}", evidence_to_text(wag_evidence))
        )

    def _fallback_answer(self, reason: str) -> Dict[str, Any]:
        return {
            "compliance_conclusion": "The real LLM response could not be parsed into the required JSON schema. Manual review is required.",
            "supporting_evidence": [],
            "conflict_analysis": "",
            "effective_date_analysis": "",
            "human_review_required": True,
            "limitations": reason,
        }

    def _guard_citations(self, answer: Dict[str, Any], evidence: List[Dict[str, Any]]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Keep final citations restricted to retrieved evidence IDs.

        The real LLM sometimes cites document-level IDs such as EXT-026 when the
        retriever exposed chunk-level IDs such as WAG-SYN-EXT-026-C001. This
        guard maps safe unique prefixes to retrieved IDs and removes unsupported
        citation IDs. It is intentionally conservative and auditable.
        """
        if not isinstance(answer, dict):
            return answer, {"applied": False, "reason": "answer_not_dict"}

        evidence_ids = [str(ev.get("evidence_id", "")).strip() for ev in evidence if ev.get("evidence_id")]
        evidence_id_set = set(evidence_ids)
        document_id_map: Dict[str, List[str]] = {}
        for ev in evidence:
            evidence_id = str(ev.get("evidence_id", "")).strip()
            if not evidence_id:
                continue
            metadata = ev.get("metadata") or {}
            document_ids = {
                str(metadata.get("document_id", "")).strip(),
                str(metadata.get("source_document_id", "")).strip(),
                str(ev.get("document_id", "")).strip(),
            }
            for document_id in document_ids:
                if document_id:
                    document_id_map.setdefault(document_id, []).append(evidence_id)

        def candidates_for(cited_id: str) -> List[str]:
            cited_id = str(cited_id or "").strip()
            if not cited_id:
                return []
            if cited_id in evidence_id_set:
                return [cited_id]
            if cited_id in document_id_map:
                return list(dict.fromkeys(document_id_map[cited_id]))
            matches = []
            for evidence_id in evidence_ids:
                compact = evidence_id.replace("WAG-SYN-", "").replace("RAG-", "")
                if evidence_id.startswith(cited_id + "-") or compact.startswith(cited_id + "-"):
                    matches.append(evidence_id)
            return list(dict.fromkeys(matches))

        cited = answer.get("supporting_evidence", [])
        if not isinstance(cited, list):
            answer["supporting_evidence"] = []
            answer["human_review_required"] = True
            return answer, {"applied": True, "corrected": [], "removed": ["supporting_evidence_not_list"]}

        corrected = []
        removed = []
        seen = set()
        for item in cited:
            if not isinstance(item, dict):
                removed.append(str(item))
                continue
            original_id = str(item.get("evidence_id", "")).strip()
            matches = candidates_for(original_id)
            if matches:
                fixed = dict(item)
                fixed["evidence_id"] = matches[0]
                if matches[0] != original_id:
                    fixed["reason"] = (fixed.get("reason", "") + f" Citation ID normalized from {original_id}.").strip()
                if len(matches) > 1:
                    fixed["reason"] = (fixed.get("reason", "") + " Multiple retrieved chunks matched; mapped to the highest-ranked retrieved chunk.").strip()
                if fixed["evidence_id"] not in seen:
                    corrected.append(fixed)
                    seen.add(fixed["evidence_id"])
            else:
                removed.append(original_id or "blank_evidence_id")

        answer["supporting_evidence"] = corrected
        if removed:
            answer["human_review_required"] = True
            existing = str(answer.get("limitations", "")).strip()
            guard_note = "Citation guard removed unsupported evidence IDs: " + ", ".join(removed) + "."
            answer["limitations"] = f"{existing} {guard_note}".strip()

        return answer, {
            "applied": True,
            "corrected_count": len(corrected),
            "removed": removed,
        }

    def run_one(self, qa_row: Dict[str, str], model_variant: str) -> Dict[str, Any]:
        question = qa_row.get("question", "")
        rag_evidence = []
        wag_evidence = []
        query_classification = classify_query(question, qa_row)
        route = query_classification.get("route", "hybrid")

        if self.enable_rag and model_variant == "static_rag":
            rag_evidence = self._retrieve_rag(question)
        elif self.enable_rag and model_variant in {"hybrid_rag_wag", "hybrid_rag_wag_verifier"} and route in {"rag", "hybrid"}:
            rag_evidence = self._retrieve_rag(question)

        if self.enable_wag and model_variant == "wag_only":
            wag_evidence = self._retrieve_wag(question)
        elif self.enable_wag and model_variant in {"hybrid_rag_wag", "hybrid_rag_wag_verifier"} and route in {"wag", "hybrid"}:
            wag_evidence = self._retrieve_wag(question)

        ranked_evidence = rank_evidence(
            rag_evidence + wag_evidence,
            qa_row,
            enable_source_ranking=self.enable_source_ranking,
            enable_date_check=self.enable_date_check,
        )
        conflict_result = detect_conflicts(qa_row, rag_evidence, wag_evidence)
        date_result = check_effective_date(qa_row, ranked_evidence, enable_date_check=self.enable_date_check)
        verifier_enabled_for_variant = self.enable_verifier and model_variant == "hybrid_rag_wag_verifier"

        llm_metadata: Dict[str, Any] = {"mode": "mock"}
        if self.use_mock:
            answer = generate_answer(
                qa_row=qa_row,
                model_variant=model_variant,
                query_classification=query_classification,
                rag_evidence=rag_evidence,
                wag_evidence=wag_evidence,
                ranked_evidence=ranked_evidence,
                conflict_result=conflict_result,
                date_result=date_result,
                verifier_enabled=verifier_enabled_for_variant,
            )
        else:
            prompt = self._build_prompt(model_variant, question, rag_evidence, wag_evidence)
            llm_metadata = call_llm(
                prompt=prompt,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw_output = llm_metadata.get("output_text") or ""
            answer = safe_json_loads(raw_output)
            if not isinstance(answer, dict):
                answer = self._fallback_answer("Parsed response was not a JSON object.")
            elif "supporting_evidence" not in answer or "human_review_required" not in answer:
                fallback = self._fallback_answer("Parsed JSON did not match required answer fields.")
                fallback["limitations"] += " Raw output excerpt: " + raw_output[:500]
                answer = fallback
            answer, citation_guard = self._guard_citations(answer, rag_evidence + wag_evidence)
            llm_metadata["citation_guard"] = citation_guard
            if verifier_enabled_for_variant:
                answer, runtime_verifier = apply_runtime_verifier(
                    answer=answer,
                    retrieved_evidence=rag_evidence + wag_evidence,
                    conflict_result=conflict_result,
                    effective_date_result=date_result,
                )
                llm_metadata["runtime_verifier"] = runtime_verifier
            if "Output was not valid JSON" in str(answer.get("limitations", "")):
                llm_metadata["raw_output_excerpt"] = raw_output[:1000]

        return {
            "qa_id": qa_row.get("qa_id") or qa_row.get("question_id"),
            "question": question,
            "question_type": qa_row.get("question_type", ""),
            "model_variant": model_variant,
            "query_classification": query_classification,
            "rag_evidence": rag_evidence,
            "wag_evidence": wag_evidence,
            "ranked_evidence": ranked_evidence[: self.top_k],
            "conflict_result": conflict_result,
            "effective_date_result": date_result,
            "answer": answer,
            "answer_generation": {
                "mode": "mock" if self.use_mock else "real_llm",
                "model_name": self.model_name if not self.use_mock else "deterministic_mock",
                "temperature": self.temperature,
                "top_k": self.top_k,
                "response_id": llm_metadata.get("response_id"),
                "provider": llm_metadata.get("provider"),
                "status": llm_metadata.get("status"),
                "finish_reason": llm_metadata.get("finish_reason"),
                "usage_metadata": llm_metadata.get("usage_metadata"),
                "raw_output_excerpt": llm_metadata.get("raw_output_excerpt"),
                "citation_guard": llm_metadata.get("citation_guard"),
                "runtime_verifier": llm_metadata.get("runtime_verifier"),
            },
            "expected_answer": qa_row.get("expected_answer", ""),
            "expected_evidence_id": qa_row.get("expected_evidence_id", ""),
            "mapped_evidence_chunk_ids": qa_row.get("mapped_evidence_chunk_ids", ""),
            "mapped_evidence_document_ids": qa_row.get("mapped_evidence_document_ids", ""),
            "conflict_type": qa_row.get("conflict_type", ""),
            "requires_human_review": qa_row.get("requires_human_review", ""),
            "expected_human_review_flag": qa_row.get("expected_human_review_flag", ""),
            "hallucination_trap_flag": qa_row.get("hallucination_trap_flag", ""),
            "effective_date_case": qa_row.get("effective_date_case", ""),
        }
