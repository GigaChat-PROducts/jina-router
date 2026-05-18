from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from optimum.exporters.onnx.config import TextEncoderOnnxConfig
from optimum.exporters.onnx.convert import export
from optimum.utils.normalized_config import NormalizedTextConfig
from transformers import AutoTokenizer

from src.training.model_constants import MODEL_NAME
from src.training.modeling import JinaForRanking


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


def _load_tokenizer(source_dir: Path):
    try:
        return AutoTokenizer.from_pretrained(source_dir)
    except (OSError, ValueError):
        return AutoTokenizer.from_pretrained(MODEL_NAME)


def _write_model_card(output_dir: Path, source_dir: Path, onnx_path: Path, opset: int):
    readme_path = output_dir / "README.md"
    if readme_path.exists():
        return

    content = f"""---
library_name: transformers
tags:
- onnx
- text-ranking
---

# Jina reranker export

This folder was exported from `{source_dir}`.

## Artifacts

- Model weights and config in the Hugging Face format
- Tokenizer files saved with `save_pretrained`
- ONNX model at `{onnx_path.name}`

## Export details

- ONNX opset: {opset}
- Exported at: {datetime.now(timezone.utc).isoformat()}
"""
    readme_path.write_text(content)


def export_model_bundle(
    source_model_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    tokenizer_source_dir: str | Path | None = None,
    opset: int = 17,
):
    source_model_dir = Path(source_model_dir)
    output_dir = (
        Path(output_dir) if output_dir is not None else source_model_dir / "final"
    )
    tokenizer_source_dir = (
        Path(tokenizer_source_dir) if tokenizer_source_dir else source_model_dir
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    model = JinaForRanking.from_pretrained(source_model_dir, trust_remote_code=True)
    model.eval()

    tokenizer = _load_tokenizer(tokenizer_source_dir)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(output_dir)

    onnx_config = JinaRerankerOnnxConfig(model.config)
    onnx_path = output_dir / "model.onnx"
    export(model=model, config=onnx_config, output=onnx_path, opset=opset)

    metadata = {
        "source_model_dir": str(source_model_dir),
        "tokenizer_source_dir": str(tokenizer_source_dir),
        "output_dir": str(output_dir),
        "onnx_path": str(onnx_path),
        "opset": opset,
        "model_class": model.__class__.__name__,
    }
    (output_dir / "onnx_export.json").write_text(json.dumps(metadata, indent=2))
    _write_model_card(output_dir, source_model_dir, onnx_path, opset)

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Export a trained Jina reranker bundle to ONNX."
    )
    parser.add_argument(
        "source_model_dir", help="Directory containing the trained model or checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination directory for the export bundle. Defaults to <source_model_dir>/final.",
    )
    parser.add_argument(
        "--tokenizer-source-dir",
        default=None,
        help="Optional directory to load tokenizer files from. Defaults to source_model_dir.",
    )
    parser.add_argument(
        "--opset", type=int, default=17, help="ONNX opset version to export with."
    )
    args = parser.parse_args()

    export_model_bundle(
        source_model_dir=args.source_model_dir,
        output_dir=args.output_dir,
        tokenizer_source_dir=args.tokenizer_source_dir,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
