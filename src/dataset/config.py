from pydantic_settings import BaseSettings

from .product_mapping import ALLOWED_COMBINATIONS
from .schemas import Combination


class DatasetConfig(BaseSettings):
    target_size: int = 3000
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1

    combinations: list[Combination] = [
        Combination(**comb) for comb in ALLOWED_COMBINATIONS
    ]

    dropout: float = 0.2
    shuffle: bool = True
    random_state: int = 42
