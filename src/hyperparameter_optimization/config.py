from optuna import Trial
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class FloatParameter(BaseModel):
    alias: str
    low: float
    high: float
    enable: bool
    default: float | None = None

    def post_init(self):
        if not self.enable and self.default is None:
            raise ValueError(
                f"Default value must be provided for disabled parameter '{self.alias}'"
            )

    def convert(self, trial: Trial) -> float:
        return (
            trial.suggest_float(self.alias, self.low, self.high)
            if self.enable
            else (self.default or 0)
        )


class OptimizerSettings(BaseSettings):
    run_name: str = "baseline_v3"
    n_trials: int = 1000

    # optimized parameters
    task_temperature: FloatParameter = FloatParameter(
        alias="task_temperature", low=0.1, high=3.0, enable=True
    )
    product_temperature: FloatParameter = FloatParameter(
        alias="product_temperature", low=0.1, high=3.0, enable=True
    )

    base_product_multiplier: FloatParameter = FloatParameter(
        alias="base_product_multiplier", low=1.0, high=3.0, enable=True
    )
    current_product_multiplier: FloatParameter = FloatParameter(
        alias="current_product_multiplier", low=1.0, high=3.0, enable=True
    )
    future_product_multiplier: FloatParameter = FloatParameter(
        alias="future_product_multiplier", low=1.0, high=3.0, enable=False, default=1.0
    )

    document_multiplier: FloatParameter = FloatParameter(
        alias="document_multiplier", low=1.0, high=5.0, enable=True
    )


optimizer_settings = OptimizerSettings()
