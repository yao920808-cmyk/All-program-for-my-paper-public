from __future__ import annotations

from pathlib import Path

from config import DEFAULT_PATHS
from feature_mlp_suite import build_argparser as build_mlp_parser, run_feature_mlp_suite
from fusion_graphsage_suite import build_argparser as build_c_parser, run_fusion_graphsage_suite
from pairnet_suite import build_argparser as build_pairnet_parser, run_pairnet_suite
from traditional_ml_suite import build_argparser as build_trad_parser, run_traditional_suite


def main() -> None:
    trad_args = build_trad_parser().parse_args([])
    mlp_args = build_mlp_parser().parse_args([])
    pairnet_args = build_pairnet_parser().parse_args([])
    c_args = build_c_parser().parse_args([])

    print("\n=== A. Traditional ML ===")
    run_traditional_suite(trad_args)
    print("\n=== B1. Feature MLP ===")
    run_feature_mlp_suite(mlp_args)
    print("\n=== B2. PairNet ===")
    run_pairnet_suite(pairnet_args)
    print("\n=== C. Fusion / GraphSAGE comparison ===")
    run_fusion_graphsage_suite(c_args)


if __name__ == "__main__":
    main()
