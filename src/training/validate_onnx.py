import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

from src.training.modeling import JinaForRanking

MODEL_DIR = "outputs/checkpoint-50/final"
CHECKPOINT_DIR = "outputs/checkpoint-50"

DOC_TOKEN_ID = 151670

# ---------------------------
# Tokenizer
# ---------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

with open("src/training/input_example.txt") as f:
    text = f.read()

pt_inputs = tokenizer(
    text,
    return_tensors="pt",
)

np_inputs = {
    "input_ids": pt_inputs["input_ids"].numpy(),
    "attention_mask": pt_inputs["attention_mask"].numpy(),
}

# ---------------------------
# PyTorch model
# ---------------------------

torch_model = JinaForRanking.from_pretrained(
    CHECKPOINT_DIR,
    trust_remote_code=True,
)

torch_model.eval()

with torch.no_grad():
    torch_outputs = torch_model(**pt_inputs)

torch_scores = torch_outputs.scores

# ---------------------------
# ONNX model
# ---------------------------

session = ort.InferenceSession(
    f"{MODEL_DIR}/model.onnx",
    providers=["CPUExecutionProvider"],
)

onnx_outputs = session.run(
    None,
    np_inputs,
)

onnx_scores = torch.tensor(onnx_outputs[0])

# ---------------------------
# Compare
# ---------------------------

doc_mask = pt_inputs["input_ids"] == DOC_TOKEN_ID

torch_doc_scores = torch_scores[doc_mask]
onnx_doc_scores = onnx_scores[doc_mask]

print("\n=== PYTORCH DOC SCORES ===")
print(torch_doc_scores)

print("\n=== ONNX DOC SCORES ===")
print(onnx_doc_scores)

print("\n=== ABS DIFF ===")
print((torch_doc_scores - onnx_doc_scores).abs())

print("\n=== MAX ABS DIFF ===")
print((torch_doc_scores - onnx_doc_scores).abs().max())
pass