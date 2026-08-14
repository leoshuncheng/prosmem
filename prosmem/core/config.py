"""Configuration for ProsMem."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ProsMemConfig:
    # LLM API: any OpenAI-compatible chat-completions endpoint.
    # Configure via env: OPENAI_API_KEY + OPENAI_BASE_URL (see .env.example).
    llm_api_key: str = ""
    llm_base_url: str = ""

    # Optional HTTP(S) proxy for the LLM client; usually left empty.
    http_proxy: str = ""

    # Model identifiers follow the serving provider's catalog naming
    # (vendor/model convention); the embedding model runs locally via
    # sentence-transformers and needs no API key.

    # Models
    main_model: str = "deepseek/deepseek-chat-v3-0324"  # For agent reasoning + baselines
    monitor_model: str = "qwen/qwen-2.5-7b-instruct"  # PM slow-path checks; well-calibrated YES/NO

    # Embedding (local)
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Retrieval thresholds
    default_semantic_threshold: float = 0.7
    focal_discount: float = 0.92

    # Ablation flags for the ablation experiments
    enable_associative: bool = True   # Ablation (a): disable for Monitor-Only variant
    enable_monitor: bool = True       # Ablation (a): disable for Assoc-Only variant
    enable_gating: bool = True        # Ablation (b): disable for Always-On Monitor
    enable_ii: bool = True            # Ablation (c): disable implementation intention encoding
    # End-to-end / encoder robustness: when True the StrategicMonitor does NOT hard-skip
    # focal (focality>=0.7) event intentions, so a mis-inferred focality still gets the
    # LLM monitor as a fallback (rescues event non-focal A2/A3). Default False = the
    # efficient hard-gate used by the structured main table (unchanged).
    enable_focal_fallback: bool = False
    # End-to-end / encoder robustness: when True the EVENT_BASED non-focal monitor
    # prompt switches from surface "does context MATCH the cue" to a one-hop
    # "does the situation INVOLVE / LEAD TO the cue" reframe, so an auto-encoded cue
    # that surfaces indirectly in a later turn (A2/A3/D3-event) still bridges. Default
    # False = the surface-match prompt used by the structured main table (unchanged).
    enable_monitor_multihop: bool = False

    @classmethod
    def from_env(cls) -> ProsMemConfig:
        return cls(
            llm_api_key=os.environ.get("OPENAI_API_KEY", ""),
            llm_base_url=os.environ.get("OPENAI_BASE_URL", ""),
        )
