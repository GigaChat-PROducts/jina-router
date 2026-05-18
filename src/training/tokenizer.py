from pydantic_settings import BaseSettings
from transformers import AutoTokenizer

from src.training.model_constants import MODEL_NAME
from src.utils import format_docs_prompts_func


class ModelTokenizerConfig(BaseSettings):
    instruction: str | None = None
    no_thinking: bool = True
    special_token: str | None = "\n"


class ModelTokenizer:
    def __init__(self, config: ModelTokenizerConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.unk_token
            )
        self.special_tokens = {
            "query_embed_token": "<|rerank_token|>",
            "doc_embed_token": "<|embed_token|>",
        }
        self.config = config

    def save_pretrained(self, output_dir: str):
        return self.tokenizer.save_pretrained(output_dir)

    def tokenize(self, texts: list[str]):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
        )

    def format_data(
        self,
        query: str,
        docs: list[str],
        max_length: int,
    ):
        instruction = self.config.instruction
        no_thinking = self.config.no_thinking
        special_token = self.config.special_token

        formatted_input = format_docs_prompts_func(
            query=query,
            docs=docs,
            instruction=instruction,
            special_tokens=self.special_tokens,
            no_thinking=no_thinking,
        )
        inputs = self.tokenizer.encode(formatted_input)
        if len(inputs) <= max_length:
            return formatted_input
        shift = len(inputs) - max_length
        query_inputs = self.tokenizer.encode(query)
        if shift > len(query_inputs):
            raise ValueError(
                f"Query is too short to even truncate to fit in the maximum length.\n {shift=}, Query inputs length: {len(query_inputs)}"
            )
        truncated_query = self.tokenizer.decode(query_inputs[:-shift])
        if special_token is not None:
            truncated_query = truncated_query[truncated_query.find(special_token) + 1 :]
        return format_docs_prompts_func(
            query=truncated_query,
            docs=docs,
            instruction=instruction,
            special_tokens=self.special_tokens,
            no_thinking=no_thinking,
        )


if __name__ == "__main__":
    config = ModelTokenizerConfig()
    tokenizer = ModelTokenizer(config)
    val = tokenizer.tokenize(["Hey", "Hi"])
    pass
