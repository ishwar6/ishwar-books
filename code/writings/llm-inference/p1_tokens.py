"""How Qwen2.5-0.5B's tokenizer cuts text into tokens (for the first figure of Part 1).
Output: results/p1_tokens_stdout.txt"""
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
print(f"vocabulary size: {len(tok):,}")
for text in ["The cat sat on the", "What is the capital of France?", "Tokenization is unbelievably useful"]:
    ids = tok(text)["input_ids"]
    print(f"{text!r}: {len(ids)} tokens")
    for i in ids:
        print(f"   {i:6d}  {tok.decode([i])!r}")
