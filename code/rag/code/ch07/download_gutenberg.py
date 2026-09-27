"""Chapter 7 - download a few public-domain books as "noise" data.

Real corpora are big and mostly irrelevant to any given question. Three novels
(~2.4 MB of text) let us see what happens to retrieval when distractors arrive.

Run:  uv run python code/ch07/download_gutenberg.py
"""
import httpx

from ragbook import DATA_DIR

BOOKS = {
    1342: "pride-and-prejudice",
    84: "frankenstein",
    2701: "moby-dick",
}
OUT = DATA_DIR / "gutenberg"


def strip_gutenberg_boilerplate(text: str) -> str:
    """Keep only the book: Gutenberg wraps every file in a licence header/footer."""
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start != -1:
        text = text[text.find("\n", start) + 1 :]
    if end != -1:
        text = text[: text.rfind("*** END OF")]
    return text.strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for book_id, slug in BOOKS.items():
        path = OUT / f"{slug}.txt"
        if path.exists():
            print(f"have    {path.name} ({path.stat().st_size // 1024} KB)")
            continue
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        try:
            r = httpx.get(url, timeout=30, follow_redirects=True)
            r.raise_for_status()
        except httpx.HTTPError as e:
            print(f"FAILED  {slug}: {e} - skipping (the rest of the chapter works without it)")
            continue
        path.write_text(strip_gutenberg_boilerplate(r.text), encoding="utf-8")
        print(f"saved   {path.name} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
