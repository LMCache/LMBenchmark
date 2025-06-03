import os
import argparse
import requests
from transformers import AutoTokenizer

TOKEN_THRESHOLDS = [
    (8192, "8k"),
    (16384, "16k"),
    (32768, "32k"),
    (65536, "64k"),
    (131072, "128k"),
    (262144, "256k"),
    (524288, "512k"),
    (1048576, "1M"),
]

def get_token_category(token_count):
    for threshold, dirname in TOKEN_THRESHOLDS:
        if token_count <= threshold:
            return dirname
    return None

def main():
    parser = argparse.ArgumentParser(description="Download Gutenberg books and classify by token count.")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name (e.g., 'gpt2', 'Qwen/Qwen1.5-7B', 'meta-llama/Meta-Llama-3-8B')")
    parser.add_argument("--output", type=str, default="gutenberg", help="Output base directory")
    parser.add_argument("--start", type=int, default=0, help="Start book ID (inclusive)")
    parser.add_argument("--end", type=int, default=1000, help="End book ID (inclusive)")
    args = parser.parse_args()

    print(f"Loading tokenizer for model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ProjectDownloader/1.0; +https://example.com/bot)",
    }

    for book_id in range(args.start, args.end + 1):
        url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
        try:
            print(f"Downloading: {url}")
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200 and "text" in response.headers.get("Content-Type", ""):
                text = response.text
                token_count = len(tokenizer.encode(text))
                category = get_token_category(token_count)

                if category:
                    category_dir = os.path.join(args.output, category)
                    os.makedirs(category_dir, exist_ok=True)
                    output_path = os.path.join(category_dir, f"{book_id}.txt")
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"Saved: {output_path} ({token_count} tokens)")
                else:
                    print(f"Skipped {book_id}: Too many tokens ({token_count})")

            else:
                print(f"Skipped {book_id}: Not plain text or unavailable")

        except Exception as e:
            print(f"Error downloading {book_id}: {e}")

if __name__ == "__main__":
    main()
