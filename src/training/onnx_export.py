from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import HfApi
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


def _load_tokenizer(source: Path | str, *, from_hub: bool = False):
    """Load tokenizer from local path or HF Hub."""
    try:
        if from_hub:
            return AutoTokenizer.from_pretrained(str(source))
        return AutoTokenizer.from_pretrained(Path(source))
    except (OSError, ValueError):
        # Fallback to default MODEL_NAME if local load fails
        if not from_hub:
            return AutoTokenizer.from_pretrained(MODEL_NAME)
        raise


def _write_model_card(output_dir: Path, source: str | Path, onnx_path: Path, opset: int, *, from_hub: bool = False):
    readme_path = output_dir / "README.md"
    if readme_path.exists():
        return

    source_label = f"HF Hub `{source}`" if from_hub else f"`{source}`"
    content = f"""---
library_name: transformers
tags:
- onnx
- text-ranking
---

# Jina reranker export

This folder was exported from {source_label}.

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
    source_model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    hf_model_id: str | None = None,
    tokenizer_source: str | Path | None = None,
    opset: int = 18,
    push_to_hf: str | None = None,
    skip_onnx_export: bool = False,
):
    """
    Export a Jina reranker bundle to ONNX.
    
    Args:
        source_model_dir: Local directory with trained model (mutually exclusive with hf_model_id)
        hf_model_id: Hugging Face model ID to load from hub (e.g., 'jinaai/jina-reranker-v1-base-en')
        output_dir: Destination directory for export bundle
        tokenizer_source: Optional directory or HF ID for tokenizer
        opset: ONNX opset version
        push_to_hf: HF repo ID to push the bundle to
        skip_onnx_export: Skip ONNX conversion, use existing model.onnx
    """
    # Resolve source: either local path or HF hub ID
    if hf_model_id:
        source_identifier = hf_model_id
        from_hub = True
        model_path_for_save = output_dir  # Save artifacts to output when loading from hub
    elif source_model_dir:
        source_identifier = Path(source_model_dir)
        from_hub = False
        model_path_for_save = source_identifier
    else:
        raise ValueError("Either source_model_dir or hf_model_id must be provided")

    output_dir = Path(output_dir) if output_dir is not None else (
        source_identifier / "final" if isinstance(source_identifier, Path) else Path("exported") / source_identifier.split("/")[-1]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "model.onnx"

    if skip_onnx_export:
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}. "
                "Cannot skip export without an existing ONNX file."
            )
        print(f"⏭️ Skipping ONNX export. Using existing model at {onnx_path}")
    else:
        print("🔄 Starting ONNX export...")
        
        # Load model & tokenizer
        model = JinaForRanking.from_pretrained(
            source_identifier if from_hub else source_model_dir,
            trust_remote_code=True
        )
        model.eval()
        
        tokenizer = _load_tokenizer(
            tokenizer_source or source_identifier,
            from_hub=from_hub or (tokenizer_source and str(tokenizer_source) not in [".", "./"])
        )

        # CRITICAL: Export on CPU even for GPU inference.
        model = model.cpu()

        # Warm-up to initialize lazy buffers
        try:
            with open("src/training/input_example.txt") as f:
                input_example = f.read()
        except FileNotFoundError:
            # Fallback example if file missing
            input_example = "What is the capital of France? [SEP] Paris is the capital of France."
            
        dummy = tokenizer(input_example, return_tensors="pt")
        with torch.no_grad():
            _ = model(**dummy)

        # Save artifacts locally
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.save_pretrained(output_dir)

        # Configure & export
        onnx_config = JinaRerankerOnnxConfig(model.config)
        
        export(
            model=model,
            config=onnx_config,
            output=onnx_path,
            opset=opset,
        )

        metadata = {
            "source_model_dir": str(source_model_dir) if source_model_dir else None,
            "hf_model_id": hf_model_id,
            "tokenizer_source": str(tokenizer_source) if tokenizer_source else None,
            "output_dir": str(output_dir),
            "onnx_path": str(onnx_path),
            "opset": opset,
            "model_class": model.__class__.__name__,
            "loaded_from_hub": from_hub,
        }
        (output_dir / "onnx_export.json").write_text(json.dumps(metadata, indent=2))
        _write_model_card(output_dir, source_identifier, onnx_path, opset, from_hub=from_hub)
        print(f"✅ ONNX export complete: {onnx_path}")

    # 🚀 Push to Hugging Face Hub if requested
    if push_to_hf:
        print(f"\n🌐 Pushing bundle to Hugging Face Hub: {push_to_hf}")
        api = HfApi()
        api.create_repo(repo_id=push_to_hf, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=push_to_hf,
            repo_type="model",
            commit_message=f"Upload ONNX exported Jina reranker bundle (from {'HF Hub' if from_hub else 'local checkpoint'})",
        )
        print(f"✅ Successfully pushed to https://huggingface.co/{push_to_hf}")
    elif not skip_onnx_export:
        print(f"💡 To push this bundle later, run with --push-to-hf <repo_id>")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Export a trained Jina reranker bundle to ONNX."
    )
    
    # Mutually exclusive group: local checkpoint OR HF model ID
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "source_model_dir", 
        nargs="?", 
        help="Directory containing the trained model or checkpoint (mutually exclusive with --hf-model-id)"
    )
    source_group.add_argument(
        "--hf-model-id",
        default=None,
        help="Hugging Face model ID to load base model from (e.g., 'jinaai/jina-reranker-v1-base-en'). Mutually exclusive with source_model_dir."
    )
    
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination directory for the export bundle. Defaults to <source_model_dir>/final or ./exported/<model_name>.",
    )
    parser.add_argument(
        "--tokenizer-source",
        default=None,
        help="Optional directory or HF model ID to load tokenizer files from. Defaults to source.",
    )
    parser.add_argument(
        "--opset", type=int, default=18, help="ONNX opset version to export with."
    )
    parser.add_argument(
        "--push-to-hf",
        default=None,
        help="Hugging Face repository ID (e.g., 'username/repo-name') to push the exported bundle to.",
    )
    parser.add_argument(
        "--skip-onnx-export",
        action="store_true",
        help="Skip ONNX conversion and push an already existing ONNX model from the output directory.",
    )
    args = parser.parse_args()

    export_model_bundle(
        source_model_dir=args.source_model_dir,
        output_dir=args.output_dir,
        hf_model_id=args.hf_model_id,
        tokenizer_source=args.tokenizer_source,
        opset=args.opset,
        push_to_hf=args.push_to_hf,
        skip_onnx_export=args.skip_onnx_export,
    )


if __name__ == "__main__":
    main()