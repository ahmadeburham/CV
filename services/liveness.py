def check_liveness(stub_mode: bool = True) -> dict:
    """
    Kaggle-safe liveness check.

    Kaggle notebooks cannot use webcam capture reliably, so the default behavior
    is a non-interactive testing bypass.
    """
    if stub_mode:
        return {
            "live": True,
            "message": "Liveness bypass enabled for notebook/testing environment.",
            "mode": "stub",
        }

    return {
        "live": False,
        "message": "Real webcam liveness is disabled in this environment.",
        "mode": "disabled",
    }
