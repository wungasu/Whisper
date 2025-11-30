import argparse
import os
import sys

from modules.utils.paths import KNOWLEDGE_BASE_DIR


def list_text_files(base_dir: str):
    for root, _, files in os.walk(base_dir):
        for name in sorted(files):
            if name.lower().endswith(".txt"):
                yield os.path.join(root, name)


def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


def main():
    parser = argparse.ArgumentParser(description="Print texts from the RAG knowledge base.")
    parser.add_argument(
        "--kb_dir",
        type=str,
        default=KNOWLEDGE_BASE_DIR,
        help="Path to the knowledge base directory.",
    )
    args = parser.parse_args()

    kb_dir = os.path.abspath(args.kb_dir)
    if not os.path.isdir(kb_dir):
        print(f"[ERROR] Knowledge base directory not found: {kb_dir}", file=sys.stderr)
        sys.exit(1)

    files = list(list_text_files(kb_dir))
    if not files:
        print(f"[INFO] No .txt files found in {kb_dir}")
        return

    print(f"[INFO] Printing {len(files)} file(s) from {kb_dir}\n")
    for path in files:
        rel_path = os.path.relpath(path, kb_dir)
        print(f"===== {rel_path} =====")
        print(read_file(path))
        print()


if __name__ == "__main__":
    main()

