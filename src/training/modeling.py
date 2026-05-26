from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3 import modeling_qwen3


@dataclass
class CausalLMOutputWithScores(CausalLMOutputWithPast):
    scores: Optional[torch.FloatTensor] = None
    query_embeds: Optional[torch.FloatTensor] = None
    doc_embeds: Optional[torch.FloatTensor] = None


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
            output_hidden_states=False,
            **kwargs,
        )

        hidden_states = outputs.logits
        batch_size, _, _ = hidden_states.shape

        projected = self.projector(hidden_states)  # [B,S,D]

        doc_mask = input_ids == self.doc_embed_token_id
        query_mask = input_ids == self.query_embed_token_id

        # [B,D]
        query_embeds = (
            projected * query_mask.unsqueeze(-1)
        ).sum(dim=1)

        query_embeds = torch.nn.functional.normalize(
            query_embeds,
            p=2,
            dim=-1,
        )

        projected = torch.nn.functional.normalize(
            projected,
            p=2,
            dim=-1,
        )

        # [B,S]
        scores = (
            projected * query_embeds.unsqueeze(1)
        ).sum(dim=-1)

        scores = scores.masked_fill(~doc_mask, -1e9)

        loss = None
        if labels is not None:
            losses = []

            for s, target, mask in zip(scores, labels, doc_mask):
                log_probs = torch.nn.functional.log_softmax(s[mask], dim=-1)
                s = s[mask].shape[0]

                loss = torch.nn.functional.kl_div(
                    log_probs,
                    target[:s],
                    reduction="sum",
                )

                losses.append(loss)

            loss = torch.stack(losses).mean()

        # Instead of: scores = [scores]; return CausalLMOutputWithScores(..., scores=scores)
        # Use this:

        return CausalLMOutputWithScores(
            loss=loss,
            logits=None,
            scores=scores,  # ✅ Fixed-shape tensor: [batch_size, max_docs]
        )
