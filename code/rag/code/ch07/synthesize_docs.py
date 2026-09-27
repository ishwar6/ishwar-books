"""Chapter 7 - synthesize more handbook-style documents with the LLM.

Why: you rarely have enough *labelled* data. Generating documents together with
question/answer pairs gives you both more corpus AND more golden questions.
Structured output (Pydantic) keeps the result machine-readable.

Run:  uv run python code/ch07/synthesize_docs.py --n 5
"""
import argparse
import re

import yaml
from pydantic import BaseModel, Field

from ragbook import DATA_DIR, get_llm

OUT = DATA_DIR / "generated"

TOPICS = [
    "Procurement and purchasing approvals",
    "Code review and branching standards for the Fleet Platform team",
    "Customer data retention and deletion",
    "Field service visit checklist for Atlas robots",
    "Internal mobility and promotion process",
    "Compass analytics: metric definitions",
    "Warehouse Wi-Fi requirements for Atlas deployments",
    "Company all-hands and communication cadence",
]


class QA(BaseModel):
    question: str
    answer: str = Field(description="one or two sentences, must be answerable from the body")


class GeneratedDoc(BaseModel):
    title: str
    body: str = Field(description="400-600 words of markdown with headings, concrete numbers, names of tools, and at least one table")
    qa_pairs: list[QA] = Field(description="exactly 3 question/answer pairs answerable from the body", min_length=3, max_length=3)


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:50]


def main(n: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    llm = get_llm().with_structured_output(GeneratedDoc)
    golden_path = OUT / "qa.yaml"
    golden = yaml.safe_load(golden_path.read_text()) if golden_path.exists() else []

    for i, topic in enumerate(TOPICS[:n], start=1):
        path = OUT / f"g{i:02d}-{slugify(topic)}.md"
        if path.exists():
            print(f"have  {path.name}")
            continue
        doc = llm.invoke(
            "You write internal policy documents for Lumora Robotics, a warehouse-robot company "
            "(products: Atlas A2 robot, Beacon fleet software, Compass analytics; tools: Hive HR, Ledger expenses, "
            "Voyage travel, Okta, Slack, Jira). Invent plausible, specific details.\n\n"
            f"Write the document: {topic}"
        )
        body = doc.body.strip()
        if body.startswith("# "):                      # models often repeat the title as an H1; keep one
            body = body.split("\n", 1)[1].strip()
        path.write_text(f"# {doc.title}\n\n{body}\n", encoding="utf-8")
        golden = [g for g in golden if not g["id"].startswith(f"g{i:02d}-")]   # regenerating a doc replaces its Q/A, never duplicates
        for j, qa in enumerate(doc.qa_pairs, start=1):
            golden.append({"id": f"g{i:02d}-{j}", "question": qa.question, "answer": qa.answer,
                           "sources": [path.stem], "answerable": True, "keywords": [], "tags": ["generated"]})
        print(f"wrote {path.name}  ({len(doc.body.split())} words, 3 QA pairs)")

    golden_path.write_text(yaml.safe_dump(golden, sort_keys=False, allow_unicode=True))
    print(f"{len(golden)} generated Q/A pairs in {golden_path.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    main(ap.parse_args().n)
