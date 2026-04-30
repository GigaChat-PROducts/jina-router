from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
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
        self.projector_dim = 512
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2, bias=False),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, self.projector_dim, bias=False),
        )
        # Fix 1: Move Identity assignment out of forward
        self.lm_head = nn.Identity()
        self.post_init()

        self.special_tokens = {
            "query_embed_token": "<|rerank_token|>",
            "doc_embed_token": "<|embed_token|>",
        }
        self.doc_embed_token_id = 151670
        self.query_embed_token_id = 151671

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,  # Accept other args but don't use them
    ):
        # 1. Get hidden states from the base Qwen model
        # We call super(modeling_qwen3.Qwen3ForCausalLM, self).forward to bypass
        # any parent logic that might be messing with the outputs
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        hidden_states = outputs.hidden_states[-1]
        batch_size = hidden_states.size(0)

        # 2. Mask-based extraction (The ONNX-friendly way)
        query_mask = (input_ids == self.query_embed_token_id).float()
        doc_mask = (input_ids == self.doc_embed_token_id).float()

        # Query: [batch, 1, dim]
        query_embeds = (hidden_states * query_mask.unsqueeze(-1)).sum(
            dim=1, keepdim=True
        )
        query_embeds = self.projector(query_embeds)

        # Docs: [batch, seq, dim] -> Project all, then mask
        doc_embeds = self.projector(hidden_states)

        # 3. Normalized Cosine Similarity
        query_norm = query_embeds / (
            query_embeds.norm(p=2, dim=-1, keepdim=True) + 1e-8
        )
        doc_norm = doc_embeds / (doc_embeds.norm(p=2, dim=-1, keepdim=True) + 1e-8)

        # [batch, seq]
        all_scores = (query_norm * doc_norm).sum(dim=-1)

        # 4. Gather only the doc token positions
        # This keeps the output shape [batch, num_docs]
        doc_indices = torch.where(input_ids == self.doc_embed_token_id)
        final_scores = all_scores[doc_indices].view(batch_size, -1)

        # 5. Return the same object as the original model
        return CausalLMOutputWithScores(
            scores=final_scores,
            logits=None,
            loss=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
