"""Device selection utilities."""

import torch


VALID_DEVICE_PREFERENCES = {"auto", "mps", "cuda", "cpu"}


def select_device(preferred: str = "auto", *, allow_fallback: bool = True) -> torch.device:
    """Select a PyTorch device, preferring MPS, then CUDA, then CPU."""
    normalized = preferred.lower()
    if normalized not in VALID_DEVICE_PREFERENCES:
        raise ValueError(f"Unsupported device preference: {preferred}")

    if normalized == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if normalized == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if normalized == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")

    if allow_fallback:
        return torch.device("cpu")

    raise RuntimeError(f"Requested device is unavailable: {preferred}")


get_device = select_device
