import os
from typing import Dict


def check_liveness(stub_mode: bool = False, environment: str = "local") -> Dict:
    """Modular liveness handler with notebook/kaggle-safe bypass mode."""
    if stub_mode or environment.lower() in {"notebook", "kaggle"}:
        return {
            "live": True,
            "mode": "bypass",
            "message": "Liveness bypass enabled for notebook/Kaggle mode.",
        }

    # Placeholder for a real liveness provider.
    provider = os.getenv("LIVENESS_PROVIDER", "disabled")
    if provider == "disabled":
        return {
            "live": False,
            "mode": "disabled",
            "message": "Real liveness provider is not configured.",
        }

    return {
        "live": False,
        "mode": "error",
        "message": f"Unsupported liveness provider: {provider}",
    }
