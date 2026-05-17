from enum import Enum

from pydantic import BaseModel, model_validator, ConfigDict


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

class ItemSource(Enum):
    BASE = "base"
    AUGMENTED = "augmented"


class Distribution(BaseModel):
    name: str
    probability: float


class DatasetItem(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    item_id: str
    item_class: ItemClass
    item_source: ItemSource
    base_product: str
    products: list[str]
    dialog: list[str]
    gt_task_distribution: list[Distribution] | None = None
    gt_product_distribution: list[Distribution] | None = None
    gt_product_distribution_with_context: list[Distribution] | None = None
    current_product: str | None = None
    metadata: dict

    @model_validator(mode="after")
    def validate_fields(self):
        def is_close(a, b):
            return abs(a - b) < 0.05
        all_products = set([self.base_product] + self.products)
        if self.current_product is not None and self.current_product not in all_products:
            raise ValueError(f"Current product is not present: {self.current_product}")
        if self.gt_product_distribution is None or self.gt_task_distribution is None or self.gt_product_distribution_with_context is None:
            return self
        curr = 0
        if len(self.gt_task_distribution) != 2:
            raise ValueError(f"Len gt task")
        for distrib in self.gt_task_distribution:
            if distrib.name not in ["documents", "best_practices"]:
                raise ValueError(f"{distrib.name=}")
            curr += distrib.probability
        if not is_close(curr, 1):
            raise ValueError(f"{curr=}")
        
        for source in [self.gt_product_distribution, self.gt_product_distribution_with_context]:
            if len(source) != len(all_products):
                raise ValueError(f"Shape mismatch")
            curr = 0
            for distrib in source:
                if distrib.name not in all_products:
                    raise ValueError(f"Wrong name")
                curr += distrib.probability
            if not is_close(curr, 1):
                raise ValueError(f"Probs wrong")
        return self