from pydantic import BaseModel


class Reranker(BaseModel):
    def rerank(self, query: str, candidates: list[str]) -> list[str]:
        """Reranks the candidates based on the query."""
        raise NotImplementedError("Reranking not implemented.")
