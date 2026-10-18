import os
import sys

# Modern Hugging Face Hub (v1.0+) uses native Xet-core for high-performance chunked transfers:
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

# Legacy fallback for older huggingface_hub environments using hf-transfer:
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("❌ 'huggingface_hub' is not installed.")
    print("Run with uv: uv run --with huggingface_hub python scripts/download_qwen_mlx.py")
    print("Or install via pip: pip install -U huggingface_hub")
    sys.exit(1)


def download_model(local_dir: str | None = None) -> str:
    repo_id = "chimingw/Qwen3.8-27B-Uncensored-OrcaRouter-MLX-4bit"
    print(f"🚀 Starting high-performance download for: {repo_id}...")

    kwargs = {
        "repo_id": repo_id,
        "repo_type": "model",
        "ignore_patterns": ["*.msgpack", "*.h5", "*.bin"],  # Ignore legacy weights
        "max_workers": 8,  # Parallel download workers
    }
    if local_dir:
        kwargs["local_dir"] = local_dir

    cache_path = snapshot_download(**kwargs)
    print(f"✅ Download complete! Files located at:\n   {cache_path}")
    return cache_path


if __name__ == "__main__":
    download_model()
