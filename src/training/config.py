from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingConfig(BaseSettings):
    """Configuration for HF Trainer. Non-secret defaults live here; secrets via env."""

    output_dir: str = Field(default="./outputs")
    num_train_epochs: int = Field(default=3)
    per_device_train_batch_size: int = Field(default=1)
    per_device_eval_batch_size: int = Field(default=1)
    learning_rate: float = Field(default=5e-5)
    warmup_steps: int = Field(default=20)
    weight_decay: float = Field(default=0.01)
    max_grad_norm: float = Field(default=1.0)
    gradient_accumulation_steps: int = Field(default=1)
    gradient_checkpointing: bool = Field(default=False)

    # Logging / saving
    eval_strategy: str = Field(default="steps")
    eval_steps: int = Field(default=50)
    logging_steps: int = Field(default=10)
    save_steps: int = Field(default=50)
    save_total_limit: int = Field(default=1)

    # MLflow defaults; tracking server is already running locally
    mlflow_tracking_uri: str = Field(default="http://localhost:5600")
    mlflow_experiment_name: str = Field(default="jina-router-training")

    # Misc
    # TODO CLEAR DATASET and multihead support
    # TODO single item length
    seed: int = Field(default=42)
    bf16: bool = Field(default=True)

    model_config = SettingsConfigDict(
        env_prefix="TRAINING__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_training_arguments(self):
        from transformers.training_args import TrainingArguments

        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_checkpointing=self.gradient_checkpointing,
            eval_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            seed=self.seed,
            bf16=self.bf16,
            report_to=["mlflow"],
            run_name="jina-reranker",
            eval_on_start=False,
        )
