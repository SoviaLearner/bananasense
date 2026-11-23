import argparse
from pathlib import Path

from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Upload BananaSense artifacts ke Hugging Face")
    parser.add_argument("--repo-id", required=True, help="Contoh: username/bananasense-cnn")
    parser.add_argument("--repo-type", choices=["model", "dataset"], default="model")
    parser.add_argument("--path", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--token", required=True)
    args = parser.parse_args()

    if not args.path.exists():
        raise SystemExit(f"Folder tidak ditemukan: {args.path}")
    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, exist_ok=True)
    api.upload_folder(
        folder_path=str(args.path),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        commit_message="Upload BananaSense artifacts",
    )
    print(f"Berhasil upload {args.path} ke https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
