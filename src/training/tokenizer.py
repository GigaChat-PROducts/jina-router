import torch
from pydantic_settings import BaseSettings
from transformers import AutoTokenizer

from src.training.schemas import TokenizerOutput
from src.utils import format_docs_prompts_func


class ModelTokenizerConfig(BaseSettings):
    max_length: int = 2560
    instruction: str | None = None
    no_thinking: bool = True
    special_token: str | None = "\n"


class ModelTokenizer:
    def __init__(self, config: ModelTokenizerConfig):
        self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-reranker-v3")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.unk_token
            )
        self.special_tokens = {
            "query_embed_token": "<|rerank_token|>",
            "doc_embed_token": "<|embed_token|>",
        }
        self.config = config

    def tokenize(self, text: str) -> TokenizerOutput:
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            max_length=self.config.max_length,
        )

        input_ids: torch.LongTensor = inputs["input_ids"]
        attention_mask: torch.Tensor = inputs["attention_mask"]

        if input_ids.shape[1] > self.config.max_length:
            raise ValueError(
                f"Tokenized input exceeds maximum length of {self.config.max_length} tokens"
            )

        return TokenizerOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def format_and_tokenize(
        self,
        query: str,
        docs: list[str],
    ):
        instruction = self.config.instruction
        no_thinking = self.config.no_thinking
        special_token = self.config.special_token
        max_length = self.config.max_length

        formatted_input = format_docs_prompts_func(
            query=query,
            docs=docs,
            instruction=instruction,
            special_tokens=self.special_tokens,
            no_thinking=no_thinking,
        )
        inputs = self.tokenizer.encode(formatted_input)
        if len(inputs) <= max_length:
            return self.tokenize(formatted_input)
        shift = len(inputs) - max_length
        query_inputs = self.tokenizer.encode(query)
        if shift >= len(query_inputs):
            raise ValueError(
                "Query is too short to even truncate to fit in the maximum length"
            )
        truncated_query = self.tokenizer.decode(query_inputs[:-shift])
        if special_token is not None:
            truncated_query = truncated_query[truncated_query.find(special_token) + 1 :]
        formatted_input = format_docs_prompts_func(
            query=truncated_query,
            docs=docs,
            instruction=instruction,
            special_tokens=self.special_tokens,
            no_thinking=no_thinking,
        )
        return self.tokenize(formatted_input)
