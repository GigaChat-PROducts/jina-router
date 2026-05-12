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
        # TODO TOKENIZER
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

    def truncate(self, text: str, max_len: int, special_token: str | None = None) -> str:
        inputs = self.tokenizer.encode(text)
        if len(inputs) <= max_len:
            return text
        
        inputs = inputs[-max_len:]
        res = self.tokenizer.decode(inputs)
        if special_token is None:
            return res
        return res[res.find(special_token) + 1:]


if __name__ == "__main__":
    tokenizer = ModelTokenizer()
    print(tokenizer.truncate(text="Hey, yo \n jkjsd \n jdksda \n", max_len=10, special_token=None))
