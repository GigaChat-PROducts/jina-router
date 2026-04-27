from enum import Enum

from pydantic import BaseModel


class Combination(BaseModel):
    id: str
    name: str
    main_product: str
    content: list[str]


class D30Item(BaseModel):
    item_id: str
    dialog_id: str
    products: list[str]
    dialog: list[str]
    dialog_type: str


class ItemClass(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Distribution(BaseModel):
    name: str
    probability: float


class DatasetItem(BaseModel):
    item_id: str
    item_class: ItemClass
    base_product: str
    products: list[str]
    dialog: list[str]
    gt_task_distribution: list[Distribution] | None = None
    gt_product_distribution: list[Distribution] | None = None
    current_product: str | None = None
    metadata: dict
