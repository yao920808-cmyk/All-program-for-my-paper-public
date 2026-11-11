"""LLM client abstraction for final non-mock experiments."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Dict, Any, List


def deterministic_mock_response(
    question: str,
    model_variant: str,
    rag_evidence: List[Dict[str, Any]] | None = None,
    wag_evidence: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    A deterministic mock output used only for pipeline testing.
    Replace this with real API calls for the final experiment.
    """
    rag_evidence = rag_evidence or []
    wag_evidence = wag_evidence or []
    evidence = []
    if model_variant in {"static_rag", "hybrid_rag_wag", "hybrid_rag_wag_verifier"}:
        evidence.extend(rag_evidence[:2])
    if model_variant in {"wag_only", "hybrid_rag_wag", "hybrid_rag_wag_verifier"}:
        evidence.extend(wag_evidence[:2])

    q_lower = question.lower()
    conflict_terms = ["conflict", "internal policy", "external", "newer", "outdated"]
    date_terms = ["effective date", "after", "before", "applies", "fiscal years ending"]
    hallucination_terms = ["asc 999", "pcaob rule 0000", "fasb asu 2099", "imaginary", "nonexistent"]

    conflict_detected = any(t in q_lower for t in conflict_terms)
    effective_date = any(t in q_lower for t in date_terms)
    trap = any(t in q_lower for t in hallucination_terms)

    if trap:
        conclusion = "The requested rule or standard cannot be verified from the available evidence. Manual review is required."
        human_review = True
    elif conflict_detected and model_variant in {"hybrid_rag_wag", "hybrid_rag_wag_verifier"}:
        conclusion = "A potential conflict exists between internal policy and external evidence. Manual compliance review is required before final conclusion."
        human_review = True
    elif evidence:
        conclusion = "The conclusion should be based on the cited evidence. No unsupported accounting standard should be invented."
        human_review = conflict_detected or effective_date
    else:
        conclusion = "Insufficient evidence is available. Manual review is required."
        human_review = True

    return {
        "compliance_conclusion": conclusion,
        "supporting_evidence": [
            {"evidence_id": ev.get("evidence_id"), "source_type": ev.get("source_type"), "reason": "Retrieved as relevant evidence."}
            for ev in evidence
        ],
        "conflict_analysis": "Potential conflict identified." if conflict_detected else "No explicit conflict identified from available evidence.",
        "effective_date_analysis": "Effective date should be checked." if effective_date else "No effective-date issue identified.",
        "human_review_required": human_review,
        "limitations": "Mock response for pipeline testing. Use a real LLM API for final reported results."
    }


ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "compliance_conclusion": {"type": "string"},
        "supporting_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "evidence_id": {"type": "string"},
                    "source_type": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["evidence_id", "source_type", "reason"],
            },
        },
        "conflict_analysis": {"type": "string"},
        "effective_date_analysis": {"type": "string"},
        "human_review_required": {"type": "boolean"},
        "limitations": {"type": "string"},
    },
    "required": [
        "compliance_conclusion",
        "supporting_evidence",
        "conflict_analysis",
        "effective_date_analysis",
        "human_review_required",
        "limitations",
    ],
}


def _extract_output_text(data: Dict[str, Any]) -> str:
    if data.get("output_text"):
        return str(data["output_text"])
    parts = []
    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text" and content.get("text") is not None:
                parts.append(str(content["text"]))
    return "\n".join(parts)


def _extract_chat_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text") is not None:
                parts.append(str(item.get("text")))
        return "\n".join(parts)
    return ""


def _extract_gemini_text(data: Dict[str, Any]) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    text_parts = []
    for part in parts:
        if isinstance(part, dict) and part.get("text") is not None:
            text_parts.append(str(part["text"]))
    return "\n".join(text_parts)


def _redact_secrets(text: str) -> str:
    text = re.sub(r"sk-[A-Za-z0-9_\-]{12,}", "sk-***REDACTED***", text)
    return re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "AIza***REDACTED***", text)


def _retry_delay_seconds(detail: str, headers: Any = None) -> float:
    retry_after = None
    if headers is not None:
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    patterns = [
        r'"retryDelay"\s*:\s*"([0-9.]+)s"',
        r"retry in\s+([0-9.]+)s",
        r"retry in\s+([0-9.]+)\s+seconds",
    ]
    for pattern in patterns:
        match = re.search(pattern, detail, flags=re.I)
        if match:
            return float(match.group(1))
    return 30.0


def _openrouter_model_name(model_name: str) -> str:
    if "/" in model_name:
        return model_name
    return f"openai/{model_name}"


def _is_openrouter(api_key: str, base: str) -> bool:
    return api_key.startswith("sk-or-") or "openrouter.ai" in base.lower()


def _is_gemini(model_name: str, base: str) -> bool:
    return model_name.startswith("gemini-") or "generativelanguage.googleapis.com" in base.lower()


def _request_json(url: str, payload: Dict[str, Any], api_key: str, headers: Dict[str, str] | None = None) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if headers:
        req_headers.update(headers)
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "5"))
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = _redact_secrets(exc.read().decode("utf-8", errors="replace"))
            if exc.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                delay = _retry_delay_seconds(detail, exc.headers) + 1.0
                print(f"LLM API HTTP {exc.code}; retrying in {delay:.1f}s ({attempt + 1}/{max_retries})", flush=True)
                time.sleep(delay)
                continue
            raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt < max_retries:
                delay = min(60.0, 2.0 ** attempt)
                print(f"LLM API request failed: {exc.reason}; retrying in {delay:.1f}s ({attempt + 1}/{max_retries})", flush=True)
                time.sleep(delay)
                continue
            raise RuntimeError(f"LLM API request failed: {exc.reason}") from exc
    raise RuntimeError("LLM API request failed after retries.")


def _request_json_with_headers(url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "5"))
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = _redact_secrets(exc.read().decode("utf-8", errors="replace"))
            if exc.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                delay = _retry_delay_seconds(detail, exc.headers) + 1.0
                print(f"LLM API HTTP {exc.code}; retrying in {delay:.1f}s ({attempt + 1}/{max_retries})", flush=True)
                time.sleep(delay)
                continue
            raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt < max_retries:
                delay = min(60.0, 2.0 ** attempt)
                print(f"LLM API request failed: {exc.reason}; retrying in {delay:.1f}s ({attempt + 1}/{max_retries})", flush=True)
                time.sleep(delay)
                continue
            raise RuntimeError(f"LLM API request failed: {exc.reason}") from exc
    raise RuntimeError("LLM API request failed after retries.")


def _call_openrouter_chat(prompt: str, model_name: str, temperature: float, max_tokens: int, api_key: str, base: str) -> Dict[str, Any]:
    url = f"{base.rstrip('/')}/chat/completions"
    model = _openrouter_model_name(model_name)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "compliance_answer",
                "strict": True,
                "schema": ANSWER_SCHEMA,
            },
        },
    }
    data = _request_json(
        url,
        payload,
        api_key,
        headers={
            "HTTP-Referer": "https://localhost/rag-wag-compliance-audit",
            "X-Title": "RAG-WAG Compliance Auditing Experiment",
        },
    )
    return {
        "provider": "openrouter_chat_completions",
        "model": data.get("model") or model,
        "response_id": data.get("id"),
        "status": "completed",
        "output_text": _extract_chat_text(data),
    }


def _call_openai_responses(prompt: str, model_name: str, temperature: float, max_tokens: int, api_key: str, base: str) -> Dict[str, Any]:
    url = f"{base.rstrip('/')}/responses"
    payload = {
        "model": model_name,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "compliance_answer",
                "strict": True,
                "schema": ANSWER_SCHEMA,
            }
        },
    }
    data = _request_json(url, payload, api_key)
    return {
        "provider": "openai_responses_api",
        "model": model_name,
        "response_id": data.get("id"),
        "status": data.get("status"),
        "output_text": _extract_output_text(data),
    }


def _call_gemini_generate_content(prompt: str, model_name: str, temperature: float, max_tokens: int, api_key: str, base: str) -> Dict[str, Any]:
    url = f"{base.rstrip('/')}/models/{model_name}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            "responseJsonSchema": ANSWER_SCHEMA,
        },
    }
    data = _request_json_with_headers(
        url,
        payload,
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
    )
    candidates = data.get("candidates") or []
    first = candidates[0] if candidates else {}
    usage = data.get("usageMetadata") or {}
    return {
        "provider": "google_gemini_generate_content",
        "model": model_name,
        "response_id": data.get("responseId") or data.get("modelVersion"),
        "status": "completed" if data.get("candidates") else "empty",
        "finish_reason": first.get("finishReason"),
        "usage_metadata": usage,
        "output_text": _extract_gemini_text(data),
    }


def call_llm(
    prompt: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 1200,
    api_base: str | None = None,
) -> Dict[str, Any]:
    """Call a supported LLM API and return parsed metadata plus text."""
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or gemini_key
    if not api_key:
        raise RuntimeError("No OPENAI_API_KEY, LLM_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY found. Use --mock or set an API key before running a real LLM experiment.")

    if gemini_key and (model_name.startswith("gemini-") or not (os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"))):
        default_base = "https://generativelanguage.googleapis.com/v1beta"
    else:
        default_base = "https://openrouter.ai/api/v1" if api_key.startswith("sk-or-") else "https://api.openai.com/v1"
    base = (api_base or os.getenv("LLM_API_BASE") or default_base).rstrip("/")

    if _is_gemini(model_name, base):
        if not gemini_key:
            raise RuntimeError("Gemini model requested, but GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
        return _call_gemini_generate_content(prompt, model_name, temperature, max_tokens, gemini_key, base)
    if _is_openrouter(api_key, base):
        return _call_openrouter_chat(prompt, model_name, temperature, max_tokens, api_key, base)
    return _call_openai_responses(prompt, model_name, temperature, max_tokens, api_key, base)
