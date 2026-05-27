from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingConfig(BaseSettings):
    """Configuration for HF Trainer. Non-secret defaults live here; secrets via env."""

    output_dir: str = Field(default="./outputs")
    final_dir_name: str = Field(default="final")
    export_onnx: bool = Field(default=False)
    onnx_opset: int = Field(default=18)

    # LoRA defaults tuned for decoder-only transformers.
    lora_r: int = Field(default=8)
    lora_alpha: int = Field(default=32)
    lora_dropout: float = Field(default=0.05)
    lora_bias: Literal["none", "all", "lora_only"] = Field(default="none")
    lora_use_rslora: bool = Field(default=True)
    lora_target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    num_train_epochs: int = Field(default=1)
    per_device_train_batch_size: int = Field(default=1)
    per_device_eval_batch_size: int = Field(default=1)
    learning_rate: float = Field(default=5e-5)
    weight_decay: float = Field(default=0.01)
    max_grad_norm: float = Field(default=1.0)
    gradient_accumulation_steps: int = Field(default=1)
    gradient_checkpointing: bool = Field(default=False)

    lr_scheduler_type: str = Field(default="cosine")
    warmup_steps: int | float = Field(default=0.05)

    # Logging / saving
    eval_strategy: str = Field(default="steps")
    eval_steps: int | float = Field(default=0.05)
    logging_steps: int = Field(default=1)
    save_steps: int | float = Field(default=0.05)
    save_total_limit: int = Field(default=2)

    # MLflow tracking server URL; set via TRAINING__MLFLOW_TRACKING_URI when needed
    mlflow_tracking_uri: str = Field(default="http://localhost:5600")
    mlflow_experiment_name: str = Field(default="jina-router-training")

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
            lr_scheduler_type=self.lr_scheduler_type,
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
            eval_on_start=True,
        )
