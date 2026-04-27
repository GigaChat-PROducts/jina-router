import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.domain.rag.runtime.quota.schemas import QuotaTask
from src.domain.rag.runtime.retrieval.variants import RetrieverName


class ModelBaseUrlEnum(str, Enum):
    OPENAI_PROXYAPI = "https://api.proxyapi.ru/openai/v1"
    LOCALHOST = "http://localhost:8090/v1"


class ExportableSettings(BaseSettings):
    """
    Базовый класс конфигурации.
    Дампит настройки в JSON только если они изменились.
    Использует типы данных для определения секретов (SecretStr).
    """

    _DUMP_FOLDER: Path = Path("dvc_configs")

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)

        # Можно добавить флаг, чтобы отключать дамп в проде, если нужно
        if os.getenv("ALLOW_CONFIG_DUMP", "True").lower() == "true":
            self._dump_config()

    def _dump_config(self):
        self._DUMP_FOLDER.mkdir(exist_ok=True, parents=True)
        file_path = self._DUMP_FOLDER / f"{self.__class__.__name__}.json"

        # 1. Генерируем будущий контент файла в памяти
        config_dict = self.model_dump(mode="python")

        try:
            new_json_content = json.dumps(
                config_dict,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                default=self._json_serializer,
            )
        except Exception as e:
            print(f"❌ Error serializing config {self.__class__.__name__}: {e}")
            return

        # 2. Проверяем, нужно ли перезаписывать
        if self._file_needs_update(file_path, new_json_content):
            try:
                file_path.write_text(new_json_content, encoding="utf-8")
                print(f"💾 Config updated: {file_path}")
            except IOError as e:
                print(f"⚠️ Failed to write config {file_path}: {e}")
        else:
            print(f"✅ Config `{file_path}` didn't change")

    def _file_needs_update(self, path: Path, new_content: str) -> bool:
        """
        Возвращает True, если файла нет или его содержимое отличается.
        """
        if not path.exists():
            return True

        try:
            current_content = path.read_text(encoding="utf-8")
            return current_content != new_content
        except Exception:
            return True

    @staticmethod
    def _json_serializer(obj: Any) -> str:
        if isinstance(obj, SecretStr):
            return "<SECRET>"
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "value"):  # Enum
            return obj.value
        return str(obj)


class LogsSettings(ExportableSettings):
    level: str = "DEBUG"
    colorize: bool = True

    model_config = SettingsConfigDict(
        env_prefix="LOGGER__",
        env_nested_max_split=1,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class RAGSettings(ExportableSettings):
    """Настройки RAG-пайплайна."""

    frida_endpoint: str = Field(
        default="http://91.211.217.36:8022/api/v1/embedder",
        description="URL эндпоинта FRIDA-сервиса для получения эмбеддингов",
    )
    frida_max_retries: int = Field(
        default=1,
        description="Максимальное количество повторных попыток запроса к FRIDA",
    )
    frida_timeout: int = Field(
        default=30,
        description="Таймаут одного запроса к FRIDA в секундах",
    )
    frida_batch_size: int = Field(
        default=4,
        description="Размер батча текстов для одного запроса эмбеддингов",
    )
    reranker_endpoint: str = Field(
        default="http://91.211.217.36:8022/api/v1/reranker",
        description=(
            "URL reranker-сервиса. Используется и для legacy reranker-подхода "
            "из коммита 11fa99a, и для cross-encoder retriever-а."
        ),
    )
    cross_encoder_model: str = Field(
        default="",
        description=(
            "Имя модели для model-aware cross-encoder backend-а. "
            "Пустое значение сохраняет legacy-контракт reranker-а из 11fa99a "
            "без явного выбора модели."
        ),
    )
    token_counter_model: str = Field(
        default="GigaChat-2-Pro",
        description="Модель GigaChat для подсчета токенов len_tokens",
    )
    force_recalculate_len_tokens: bool = Field(
        default=False,
        description="Принудительно пересчитать len_tokens при старте",
    )
    global_token_limit: int | None = Field(
        default=10000,
        description="Общий лимит токенов для одной product-конфигурации (None — без ограничений)",
    )
    enable_quota_reranking: bool = Field(
        default=True,
        description="Включить динамическое распределение token limits между задачами и продуктами через reranker",
    )
    task_reranking_temperature: float = Field(
        default=2.0,
        description="Температура softmax при расчете task-квот reranker-ом",
    )
    product_reranking_temperature: float = Field(
        default=0.5,
        description="Температура softmax при расчете product-квот reranker-ом",
    )
    base_product_multiplier: float = Field(
        default=2,
        description="Множитель вероятности base-продукта при распределении product-квот",
    )
    current_product_multiplier: float = Field(
        default=2,
        description="Множитель вероятности current-продукта (фокус модели) при распределении product-квот",
    )
    future_product_multiplier: float = Field(
        default=1.0,
        description=(
            "Множитель вероятности future-продукта. "
            "Логика присутствует, но на текущем этапе отключена в runtime."
        ),
    )
    quota_tasks: list[QuotaTask] = Field(
        default_factory=lambda: [
            QuotaTask(name="documents", multiplier=5.0),
            QuotaTask(name="best_practices", multiplier=1.0),
        ],
        description="Задачи reranker-а и их множители перед softmax",
    )
    auto_sync_artifacts: bool = Field(
        default=True,
        description="Автоматически пересобирать RAG-артефакты при изменении исходных markdown-файлов",
    )

    # TODO: должно заполняться не именами сплиттеров, а enum-ами
    documents_splitter: str = Field(
        default="markdown_header",
        description="Имя сплиттера для knowledge.md",
    )
    best_practices_splitter: str = Field(
        default="markdown_header",
        description="Имя сплиттера для best_practices.md",
    )
    retriever_name: RetrieverName = Field(
        default=RetrieverName.SEMANTIC,
        description="Имя retriever-стратегии для ранжирования чанков",
    )
    hybrid_bm25_k1: float = Field(
        default=1.2,
        description="BM25 k1 для hybrid retriever",
    )
    hybrid_bm25_b: float = Field(
        default=0.75,
        description="BM25 b для hybrid retriever",
    )
    hybrid_rrf_k: int = Field(
        default=60,
        description="RRF k для слияния dense и BM25 ранжирования",
    )
    hybrid_dense_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Вес dense-ветки для weighted hybrid retriever; "
            "вес sparse-ветки считается как (1 - hybrid_dense_weight)"
        ),
    )
    cross_encoder_candidate_limit: int | None = Field(
        default=20,
        description="Сколько top dense-кандидатов отдавать в cross-encoder rerank (None — все)",
    )
    splitter_max_chars: int = Field(
        default=512,
        description="Максимальный размер чанка для сплиттеров",
    )
    splitter_min_words: int = Field(
        default=6,
        description="Минимальное число слов в чанке после постобработки",
    )

    model_config = SettingsConfigDict(
        env_prefix="RAG__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class Settings(ExportableSettings):
    """Конфигурация проекта через переменные окружения."""

    openai_api_key: str = ""
    openai_api_base: ModelBaseUrlEnum | str = ModelBaseUrlEnum.LOCALHOST
    openai_model: str = "GigaChat-2-Pro"  # "gpt-4o-mini"
    openai_temperature: float = 0.0
    gigachat_credentials: str = Field(
        default="",
        validation_alias="GIGACHAT_CREDENTIALS",
    )
    gigachat_scope: str = Field(
        default="GIGACHAT_API_CORP",
        validation_alias="GIGACHAT_SCOPE",
    )
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_tracing_enabled: bool = True
    langfuse_tracing_environment: str = "default"

    langsmith_api_key: str = ""
    langsmith_tracing: bool = True
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = "dialog-agent-trio"

    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @field_validator("openai_api_base", mode="before")
    @classmethod
    def coerce_enum_to_str(cls, v):
        if isinstance(v, ModelBaseUrlEnum):
            return v.value
        return v


logs_settings = LogsSettings()
logger.remove()
logger.add(
    sink=sys.stdout,
    level=logs_settings.level,
    backtrace=True,
    colorize=logs_settings.colorize,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{file}:{line}</cyan> | "
        "<cyan>{message}</cyan>"
    ),
)

rag_settings = RAGSettings()
settings = Settings()
