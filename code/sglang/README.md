# SGLang and vLLM: article experiments

Source for [LLM Inference, Part 5](https://ishwarj.com/writings/llm-inference-5-sglang-vs-vllm).
Edit `article.md`, then run `assemble.py`; the published Markdown is generated from this source plus saved SVGs.

## What was run

`experiments.py` ran on macOS arm64, Python 3.12.13, NumPy 2.5.3. It performs real CPU arithmetic and simulations, not SGLang/vLLM model serving:

- Two layers of causal attention: full computation versus prefix-KV reuse, plus an intentionally invalid changed-prefix reuse.
- Compressed radix-tree insertion and prefix matching, compared with a SHA-256 full-block index on aligned and unaligned synthetic token sequences.
- 600 randomized request checks against a brute-force longest-prefix oracle and adversarial checks for edge splitting and preceding-context identity.
- Capacity-limited, whole-prefix LRU scheduling simulation (not the SGLang scheduler).
- Memory, ideal overlap, and transfer arithmetic.

Saved outputs are in `results/experiments.json` and `results/experiments_stdout.txt`. Do not present their token counts or assumed times as GPU benchmark results. The attention model has random fixed weights, no learned language ability, two layers, width 16, and float64 arithmetic. The cache indexes omit real tensors, reference counts, eviction, adapters, multimodal namespaces, and hybrid-model constraints. Requests insert sequentially with no capacity limit in the index experiment; the separate scheduler simulation has an explicit two-prefix limit.

## Reproduce from the repository root

```bash
python3 -m venv /tmp/sglang-article-env
/tmp/sglang-article-env/bin/pip install -r code/sglang/requirements.txt
/tmp/sglang-article-env/bin/python code/sglang/experiments.py
/tmp/sglang-article-env/bin/python code/sglang/figures.py
/tmp/sglang-article-env/bin/python code/sglang/assemble.py
python3 -m unittest discover -s code/sglang -p 'test_*.py'
node code/agents/check_md.mjs content/writings/llm-inference-5-sglang-vs-vllm.md
npm run typecheck
NODE_ENV=production npm run build
```

Only NumPy is needed for the calculations. Pillow and PyMuPDF reproduce the PDF excerpt. Exact recorded versions are in `requirements.txt`; other compatible versions may yield tiny numerical differences.

## Research screenshot

```bash
curl -L --fail https://arxiv.org/pdf/2312.07104v2 -o /tmp/sglang-v2.pdf
/tmp/sglang-article-env/bin/python code/sglang/paper_shots.py /tmp/sglang-v2.pdf
```

The screenshot uses actual pixels from two small, separately labelled crops on page 4, with a highlight. `results/paper_shots.json` records PDF checksum, page, and coordinates. The script uses Arial where available and falls back to Pillow's default font for the panel labels; the source excerpt itself is always rendered from the PDF. No downloaded PDF is committed.

`results/sources.json` pins the upstream source revisions and downloaded-file checksums. The official source was inspected, not installed as a benchmark. Documentation links were checked on 2026-10-08. Recheck feature flags before using another release.

## Real-server smoke test

`probe_server.py` uses the standard library and calls `/v1/completions`. It sends two questions sharing a text prefix, records first nonempty text-chunk latency, total stream latency, response text, and usage if provided. It does not assert that a cache hit occurred. It does not measure throughput, token-level timing, or serving quality. The initial cache state is unknown.

Launch a compatible server using the article's instructions, then:

```bash
python3 code/sglang/probe_server.py --url http://127.0.0.1:30000 --model comparison-model --output /tmp/sglang-probe.json
```

Use `INFERENCE_API_KEY` only if the endpoint needs authentication. The key is never saved in the result. The optional GPU launch and live-server probe were not run for this article: the authoring machine had no NVIDIA GPU. `test_probe.py` checks the parser with synthetic SSE fixtures, including empty chunks, multi-token text chunks, usage, server errors, and truncated streams. Passing those tests is not evidence of engine compatibility or performance.

Before benchmarking, record hardware, model/tokenizer snapshots, engine versions, all flags, cache state, arrival trace, generation settings, actual output lengths, errors, latency distributions, and output quality. Run the engines separately on the same hardware.
