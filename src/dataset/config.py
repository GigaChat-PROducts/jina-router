from pydantic_settings import BaseSettings

from .product_mapping import ALLOWED_COMBINATIONS
from .schemas import Combination


class DatasetConfig(BaseSettings):
    target_size: int = 100
    train_size: float = 0.0
    val_size: float = 0.0
    test_size: float = 1.0

    combinations: list[Combination] = [
        Combination(**comb) for comb in ALLOWED_COMBINATIONS
    ]

    dropout: float = 0.2
    shuffle: bool = True
    random_state: int = 42
