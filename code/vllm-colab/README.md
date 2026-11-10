# vLLM on a free Colab GPU: notebook and article

Source for [LLM Inference, Part 8](https://ishwarj.com/writings/llm-inference-8-vllm-on-colab) and its notebook.

| File | What it is |
|---|---|
| `make_notebook.py` | Builds `public/notebooks/vllm-on-colab.ipynb` (served at `/notebooks/vllm-on-colab.ipynb`). Edit cells here, not in the `.ipynb`. |
| `article.md` | Article source. `{{R:expr}}` slots are filled from the run's results; `{{FIG:key\|caption}}` are SVG figures drawn from the same results. |
| `assemble.py` | Fills the slots, draws the figures, checks em dashes, banned words, images and ASD-STE100 (`--strict`), writes `content/writings/llm-inference-8-vllm-on-colab.md`. |
| `results/colab_run.json` | The saved run the article uses (`vllm_colab_results.json` from the notebook). |
| `results/run/` | The unzipped `vllm_colab_run.zip`: every server log and raw `vllm bench serve` file. |

## Workflow

```bash
python3 code/vllm-colab/make_notebook.py            # after editing the notebook cells
# run the notebook on a Colab T4, download vllm_colab_results.json and vllm_colab_run.zip, then:
cp ~/Downloads/vllm_colab_results.json code/vllm-colab/results/colab_run.json
unzip -o ~/Downloads/vllm_colab_run.zip -d code/vllm-colab/results/run
python3 code/vllm-colab/assemble.py --strict
```

The article's prose is written for a T4 run (`assemble.py` refuses other GPUs). After a new run, reread every
sentence that interprets a number: the slots update by themselves, the interpretation does not.

## Facts checked against vLLM v0.33.0 (commit 4ba787e, 31 October 2026)

- Startup log lines parsed by the notebook: `Using %s attention backend`, `Model loading took %s GiB memory and %f seconds`,
  `torch.compile took %.2f s in total`, `Available KV cache memory: %s GiB`, `GPU KV cache size: %s tokens, Maximum
  concurrency for %s tokens per request: %.2fx`, `Graph capturing finished in %.0f secs, took %.2f GiB`,
  `init engine (profile, create kv cache, warmup model) took %.2f s`.
- Metrics: `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`,
  `vllm:num_preemptions_total`, `vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total`.
- Defaults: prefix caching on, chunked prefill on, `--gpu-memory-utilization` 0.92. `--api-key` guards `/v1`, not
  `/health` or `/metrics`. `vllm bench serve` reads the key from `OPENAI_API_KEY` and saves `request_goodput` when
  `--goodput` is given.
- T4 (compute capability 7.5): no bfloat16 (`--dtype half`), FlashAttention and FlashInfer not selected, so attention
  uses `TRITON_ATTN` (see vLLM issue #60821).

## Testing without a GPU

The notebook's plumbing was tested end to end on macOS against a fake `vllm` CLI and a fake `nvidia-smi` that print
the real v0.33.0 log formats and serve `/health`, `/metrics` and the OpenAI routes. That checks parsing, file paths and
the results schema, not vLLM's behaviour.
