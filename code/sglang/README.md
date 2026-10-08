# SGLang and vLLM: article experiments

Source for [LLM Inference, Part 5](https://ishwarj.com/writings/llm-inference-5-sglang-vs-vllm).
Edit `article.md`, then run `assemble.py`; the published Markdown is generated from this source plus saved SVGs.

## What was run

Runs on macOS arm64 (Apple M5 Pro), Python 3.12. Nothing here runs SGLang or vLLM.

- `walkthrough.py`: the three-request trace and the two-number attention example (CPU).
- `experiments.py`: two layers of causal attention (full computation versus prefix-KV reuse, plus an intentionally invalid changed-prefix reuse), a radix tree versus a SHA-256 full-block index on aligned and unaligned synthetic workloads, 600 randomized checks against a brute-force longest-prefix oracle, and a whole-prefix LRU scheduling toy (CPU).
- `measure_prefix.py`: **real timings**. Qwen2.5-0.5B, BF16, PyTorch on the Apple GPU (`mps`). Cold prefill of prefix + suffix against warm prefill of the suffix on a cached prefix; prefix sweep 512 to 8,192 tokens, suffix sweep 16 to 1,024 tokens, one decode step at context 2,112. Median of 9 runs after 3 warm-ups. Checks that the next token and logits match.
- `simulate.py`: KV bytes per token, FLOP counts, whole-request arithmetic, memory sharing, leaf-first LRU trace, block-rounding loss, a capacity sweep (radix with whole-leaf eviction, radix trimming leaf tails as an ablation, 16- and 64-token hashed blocks with a reverse-order LRU free queue), FCFS/random/LPM scheduling and a starvation case, and the jump-forward pass count with the Qwen2.5 tokenizer. Times in the scheduling sections come from a cost model fitted to `results/measure_prefix.json` (worst fit error 30%); they are estimates for this one machine, not engine benchmarks. Requests are served one at a time: no batching, no decode, no real memory pool.
- `test_simulate.py`, `test_probe.py`: unit tests for the simulated caches and the probe parser.

Outputs: `results/*.json` and `results/*_stdout.txt`. Figures: `figures.py` writes `results/figures.json`; `assemble.py` inlines them into the article.

Screenshots: `shots_paper.py` (highlighted crops of arXiv 2312.07104v2, manifest in `results/shots_paper.json`), `paper_shots.py` (the one-sentence crop), `web_shots.sh` with `docshot.mjs` (LMSYS blog figures, vLLM and SGLang docs passages), `termshots.py` (terminal pictures of the saved logs).

## Reproduce from the repository root

```bash
python3 -m venv /tmp/sglang-article-env
/tmp/sglang-article-env/bin/pip install -r code/sglang/requirements.txt
/tmp/sglang-article-env/bin/python code/sglang/walkthrough.py
/tmp/sglang-article-env/bin/python code/sglang/experiments.py
/tmp/sglang-article-env/bin/python code/sglang/measure_prefix.py   # needs a GPU or patience
/tmp/sglang-article-env/bin/python code/sglang/simulate.py
/tmp/sglang-article-env/bin/python code/sglang/termshots.py
/tmp/sglang-article-env/bin/python code/sglang/figures.py
/tmp/sglang-article-env/bin/python code/sglang/assemble.py
python3 -m unittest discover -s code/sglang -p 'test_*.py'
node code/agents/check_md.mjs content/writings/llm-inference-5-sglang-vs-vllm.md
npm run typecheck
NODE_ENV=production npm run build
```

Only NumPy is needed for the calculations. PyMuPDF reproduces the PDF excerpt. Exact recorded versions are in `requirements.txt`; other compatible versions may yield tiny numerical differences.

## Research screenshot

```bash
curl -L --fail https://arxiv.org/pdf/2312.07104v2 -o /tmp/sglang-v2.pdf
/tmp/sglang-article-env/bin/python code/sglang/paper_shots.py /tmp/sglang-v2.pdf
```

The screenshot contains one sentence from page 4. `results/paper_shots.json` records the PDF checksum, page, crop coordinates, and highlight. Production details stay here; the article explains why the finding matters to the handbook example. No downloaded PDF is committed.

`results/sources.json` pins the upstream source revisions and downloaded-file checksums. The official source was inspected, not installed as a benchmark. Documentation links were checked on 2026-10-09. Defaults quoted in the article were read from SGLang b2cb249 and vLLM c23ca06 (8 October 2026). Recheck feature flags before using another release.

## Real-server smoke test

`probe_server.py` uses the standard library and calls `/v1/completions`. It sends two questions sharing a text prefix, records first nonempty text-chunk latency, total stream latency, response text, and usage if provided. It does not assert that a cache hit occurred. It does not measure throughput, token-level timing, or serving quality. The initial cache state is unknown.

Launch a compatible server using the article's instructions, then:

```bash
python3 code/sglang/probe_server.py --url http://127.0.0.1:30000 --model comparison-model --output /tmp/sglang-probe.json
```

Use `INFERENCE_API_KEY` only if the endpoint needs authentication. The key is never saved in the result. The optional GPU launch and live-server probe were not run for this article: the authoring machine had no NVIDIA GPU. `test_probe.py` checks the parser with synthetic SSE fixtures, including empty chunks, multi-token text chunks, usage, server errors, and truncated streams. Passing those tests is not evidence of engine compatibility or performance.

Before benchmarking, record hardware, model/tokenizer snapshots, engine versions, all flags, cache state, arrival trace, generation settings, actual output lengths, errors, latency distributions, and output quality. Run the engines separately on the same hardware.
