import argparse
import os
import re

# pipenv run python clean_transcripts.py --input-dir test-transcripts --output-dir cleaned-test-transcripts

SPEAKER_LABEL = re.compile(r"(?<![A-Za-z])([A-Z][A-Za-z.'-]+(?: [A-Z][A-Za-z.'-]+){0,3}:)")
DATE_PATTERN = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}"
)
FOOTER_MARKERS = (
    "Become a Huberman Lab Premium member",
    "Become a Member",
    "sign in to Huberman Lab Premium",
)


def default_input_dir():
    if os.path.isdir("transcripts"):
        return "transcripts"
    return "test-transcripts"


def clean_metadata(text):
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Date:"):
            match = DATE_PATTERN.search(stripped)
            if match:
                cleaned.append(f"Date: {match.group(0)}")
            else:
                value = re.sub(r"\s+", " ", stripped[5:]).strip()
                cleaned.append(f"Date: {value}")
            continue

        if stripped.startswith(("Title:", "Source:")):
            key, value = stripped.split(":", 1)
            value = re.sub(r"\s+", " ", value).strip()
            cleaned.append(f"{key}: {value}")
            continue

        cleaned.append(stripped)

    return "\n".join(line for line in cleaned if line)


def trim_footer(text):
    cut_points = [text.find(marker) for marker in FOOTER_MARKERS if marker in text]
    if not cut_points:
        return text
    return text[: min(cut_points)].rstrip()


def clean_body(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = trim_footer(text)
    text = re.sub(r"([.!?])([A-Z][A-Za-z.'-]+(?: [A-Z][A-Za-z.'-]+){0,3}:)", r"\1\n\n\2", text)
    text = re.sub(r'(["\)])([A-Z][A-Za-z.'"'"'-]+(?: [A-Z][A-Za-z.'"'"'-]+){0,3}:)', r"\1\n\n\2", text)
    text = SPEAKER_LABEL.sub(r"\n\n\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return text.strip()


def clean_transcript(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n={10,}\n", text, maxsplit=1)

    if len(parts) == 2:
        metadata, body = parts
        metadata = clean_metadata(metadata)
        body = clean_body(body)
        return f"{metadata}\n\n{'=' * 80}\n\n{body}\n"

    return f"{clean_body(text)}\n"


def clean_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as handle:
        original = handle.read()

    cleaned = clean_transcript(original)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(cleaned)


def main():
    parser = argparse.ArgumentParser(description="Clean transcript formatting into a separate directory.")
    parser.add_argument("--input-dir", default=default_input_dir(), help="Directory containing raw transcripts.")
    parser.add_argument("--output-dir", default="cleaned-transcripts", help="Directory for cleaned transcripts.")
    args = parser.parse_args()

    count = 0
    for filename in sorted(os.listdir(args.input_dir)):
        if not filename.endswith((".txt", ".md")):
            continue

        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)
        clean_file(input_path, output_path)
        count += 1

    print(f"Cleaned {count} transcript files from {args.input_dir} into {args.output_dir}")


if __name__ == "__main__":
    main()
    