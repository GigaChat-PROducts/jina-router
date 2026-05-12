from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3 import modeling_qwen3


@dataclass
class CausalLMOutputWithScores(CausalLMOutputWithPast):
    scores: Optional[torch.Tensor] = None
    loss: Optional[torch.FloatTensor] = None


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
        self.loss_func = nn.KLDivLoss(reduction="batchmean")

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
        batch_size, _, dim = hidden_states.shape

        query_embed_token_indexes = torch.eq(input_ids, self.query_embed_token_id)
        doc_embed_token_indexes = torch.eq(input_ids, self.doc_embed_token_id)

        doc_embeds = hidden_states[doc_embed_token_indexes].view(batch_size, -1, dim)
        query_embeds = hidden_states[query_embed_token_indexes].unsqueeze(1)

        doc_embeds = self.projector(doc_embeds)
        query_embeds = self.projector(query_embeds)

        query_embeds_expanded = query_embeds.expand_as(doc_embeds)
        scores = torch.nn.functional.cosine_similarity(
            doc_embeds, query_embeds_expanded, dim=-1
        ).squeeze(-1)

        loss = None
        if labels is not None:
            scores = torch.nn.functional.log_softmax(scores, dim=-1)
            loss = self.loss_func(scores, labels)

        return CausalLMOutputWithScores(
            loss=loss,
            scores=scores,
        )
