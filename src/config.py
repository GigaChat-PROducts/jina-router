from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.evaluation.quotas import QuotaTask


class RAGSettings(BaseSettings):
    reranker_endpoint: str = Field(
        default="http://91.211.217.36:8022/api/v1/reranker",
        description="URL reranker endpoint",
    )
    cross_encoder_model: str = ""
    global_token_limit: int = Field(
        default=10000,
        description="Global quota budget",
    )
    enable_quota_reranking: bool = Field(
        default=True,
        description="Toggle rerank-based quota planning",
    )
    task_reranking_temperature: float = Field(
        default=1.4349012971010282,
        description="Softmax temperature for task allocation",
    )
    product_reranking_temperature: float = Field(
        default=0.3683580146682431,
        description="Softmax temperature for product allocation",
    )
    base_product_multiplier: float = Field(
        default=1.0019673318721718,
        description="Score multiplier for base product",
    )
    current_product_multiplier: float = Field(
        default=2.999039835929017,
        description="Score multiplier for current product",
    )
    future_product_multiplier: float = Field(
        default=1.0,
        description="Score multiplier for future product",
    )
    quota_tasks: list[QuotaTask] = Field(
        default_factory=lambda: [
            QuotaTask(name="documents", multiplier=1.0047327539864788),
            QuotaTask(name="best_practices", multiplier=1.0),
        ],
        description="Task definitions and weights",
    )

    model_config = SettingsConfigDict(
        env_prefix="RAG__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


rag_settings = RAGSettings()
settings = rag_settings
