import torch
from transformers import AutoTokenizer

from src.training.schemas import TokenizerOutput


class ModelTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-reranker-v3")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.unk_token
            )
        self.special_tokens = {
            "query_embed_token": "<|rerank_token|>",
            "doc_embed_token": "<|embed_token|>",
        }

    def tokenize(self, text: str) -> TokenizerOutput:
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        )

        input_ids: torch.LongTensor = inputs["input_ids"]
        attention_mask: torch.Tensor = inputs["attention_mask"]

        return TokenizerOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
