from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers.models.qwen3 import modeling_qwen3


@dataclass
class CausalLMOutputWithScores:
    scores: list[torch.Tensor]
    loss: Optional[torch.Tensor] = None


class JinaForRanking(modeling_qwen3.Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.padding_side = "left"
        self.projector_dim = 512

        self.projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2, bias=False),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, self.projector_dim, bias=False),
        )

        self.post_init()

        self.special_tokens = {
            "query_embed_token": "<|rerank_token|>",
            "doc_embed_token": "<|embed_token|>",
        }
        self.doc_embed_token_id = 151670
        self.query_embed_token_id = 151671
        self.loss_func = nn.KLDivLoss(reduction="sum")

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithScores:
        self.lm_head = nn.Identity()

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            **kwargs,
        )

        hidden_states = outputs.hidden_states[-1]
        batch_size, _, _ = hidden_states.shape

        doc_embeds = []
        query_embeds = []

        for i in range(batch_size):
            doc_mask = input_ids[i] == self.doc_embed_token_id
            query_mask = input_ids[i] == self.query_embed_token_id

            d_embeds = hidden_states[i][doc_mask]  # [num_docs_i, dim]
            q_embed = hidden_states[i][query_mask]  # [1, dim]

            doc_embeds.append(d_embeds)
            query_embeds.append(q_embed)

        doc_embeds = [self.projector(x) for x in doc_embeds]
        query_embeds = [self.projector(x) for x in query_embeds]

        scores = []

        for q, d in zip(query_embeds, doc_embeds):
            q = q.expand_as(d)

            s = torch.nn.functional.cosine_similarity(d, q, dim=-1)

            scores.append(s)

        loss = None
        if labels is not None:
            losses = []

            for s, target in zip(scores, labels):
                log_probs = torch.nn.functional.log_softmax(s, dim=-1)

                loss = self.loss_func(
                    log_probs,
                    target,
                )

                losses.append(loss)

            loss = torch.stack(losses).mean()

        return CausalLMOutputWithScores(
            loss=loss,
            scores=scores,
        )
