"""Learning-rate scheduler construction."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_scheduler(
    optimizer: Optimizer,
    scheduler_name: str | None,
    *,
    mode: str = "max",
    factor: float = 0.1,
    patience: int = 2,
) -> ReduceLROnPlateau | None:
    """Build a scheduler from config values."""
    if scheduler_name is None or scheduler_name.lower() in {"", "none"}:
        return None

    normalized = scheduler_name.lower()
    if normalized == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")
