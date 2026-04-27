from pydantic import BaseModel
from pydantic_settings import BaseSettings


class FloatParameter(BaseModel):
    alias: str
    low: float
    high: float

    def convert(self):
        return self.alias, self.low, self.high


class OptimizerSettings(BaseSettings):
    run_name: str = "baseline"
    n_trials: int = 1000

    # optimized parameters
    task_temperature: FloatParameter = FloatParameter(
        alias="task_temperature", low=0.1, high=3.0
    )
    product_temperature: FloatParameter = FloatParameter(
        alias="product_temperature", low=0.1, high=3.0
    )

    base_product_multiplier: FloatParameter = FloatParameter(
        alias="base_product_multiplier", low=1.0, high=3.0
    )
    current_product_multiplier: FloatParameter = FloatParameter(
        alias="current_product_multiplier", low=1.0, high=3.0
    )
    future_product_multiplier: FloatParameter = FloatParameter(
        alias="future_product_multiplier", low=1.0, high=3.0
    )

    document_multiplier: FloatParameter = FloatParameter(
        alias="document_multiplier", low=1.0, high=5.0
    )


optimizer_settings = OptimizerSettings()
