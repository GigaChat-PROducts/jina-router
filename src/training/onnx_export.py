from pathlib import Path

import torch
from modeling import JinaForRanking

# Import 'export' instead of 'onnx_export_from_model'
from optimum.exporters.onnx.convert import export
from transformers import AutoConfig

path = "./models/jina_reranker"
# Fix the path joining (removed leading slash in filename)
output_path = Path(path) / "onnx" / "model.onnx"
output_path.parent.mkdir(parents=True, exist_ok=True)

# 1. Load the model using your ONNX-friendly class
model = JinaForRanking.from_pretrained(path, trust_remote_code=True)
model.eval()

# 2. Define your config
from optimum.exporters.onnx.config import TextEncoderOnnxConfig
from optimum.utils.normalized_config import NormalizedTextConfig


class JinaRerankerOnnxConfig(TextEncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    @property
    def inputs(self):
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self):
        return {"scores": {0: "batch_size", 1: "sequence_length"}}


onnx_config = JinaRerankerOnnxConfig(model.config)

# 3. Use the lower-level 'export' function
# This bypasses the TasksManager library inference entirely
export(model=model, config=onnx_config, output=output_path, opset=17)

print(f"Done! Model saved to {output_path}")
