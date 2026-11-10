"""Build the vLLM-on-Colab notebook (public/notebooks/vllm-on-colab.ipynb).

Edit the cells here, then run:  python3 code/vllm-colab/make_notebook.py
The notebook is plain JSON (nbformat 4); no extra packages are needed to build it.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "public" / "notebooks" / "vllm-on-colab.ipynb"
COLAB_URL = ("https://colab.research.google.com/github/ishwar6/ishwar-books/blob/main/"
             "public/notebooks/vllm-on-colab.ipynb")
ARTICLE_URL = "https://ishwarj.com/writings/llm-inference-8-vllm-on-colab"
VLLM_VERSION = "0.33.0"

cells = []


def md(text):
    cells.append(("markdown", text.strip("\n")))


def code(text):
    cells.append(("code", text.strip("\n")))


# ---------------------------------------------------------------- 0. title
md(f"""
# Run vLLM Like Production, on a Free Colab GPU

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

This notebook starts a real vLLM server on one Colab GPU and treats it like a production service. You send it traffic, measure it, change its settings and break it on purpose. Every number it prints comes from your own run.

It is the companion of [LLM Inference, Part 8: vLLM on a Free GPU]({ARTICLE_URL}). The article explains each result. The notebook produces them.

**What you need:** a Google account. The free T4 GPU is enough. An L4 or A100 (Colab Pro or pay as you go) runs faster and unlocks one extra experiment.

**Time:** about 45 to 60 minutes on a T4 with every experiment. Set `QUICK = True` in the settings cell to cut it to about 25 minutes.

**What it does, in order:**

| Step | What you do | What you learn |
|---|---|---|
| 1 | Check the GPU, install vLLM {VLLM_VERSION} | What the T4 can and cannot do |
| 2 | Start the server, read its startup log | Where the startup time goes, how much KV cache you got |
| 3 | Send the first requests | Time to first token and decode speed for one user |
| 4 | Check the KV cache size by hand | The memory arithmetic behind the startup log |
| 5 | Raise the number of users at the same time | Why batching gives throughput, and what each user pays |
| 6 | Raise the arrival rate past capacity | Where latency breaks, and goodput |
| 7 | Turn prefix caching off and on | What reusing a shared prompt saves |
| 8 | Give the server a small KV cache | Preemption, and why memory settings matter |
| 9 | Break it: long prompts, wrong key, a killed server | What failures look like from the client side |
| 10 | (L4 or A100 only) FP8 weights | What quantization buys on a newer GPU |
| 11 | Save everything | One JSON file and one zip with all results and logs |

> **Before you start:** open **Runtime > Change runtime type**, choose **T4 GPU** (or L4 / A100), and save. Then run the cells from top to bottom with **Shift + Enter**. Do not skip cells: later cells use the results of earlier ones.
""")

md("""
## Things to know before you run

- **The server runs in the background.** The notebook starts `vllm serve` as a separate process and talks to it over HTTP on `localhost:8000`, like any client would. The notebook never imports vLLM or PyTorch itself. So you do not need to restart the runtime after the install.
- **One server at a time.** The GPU has room for one copy of the model with its cache. The helper stops the old server before it starts a new one, and it waits until the GPU memory is free.
- **Colab sessions end.** The free tier stops idle sessions and caps the session length. If the runtime disconnects, the server and all results in memory are gone. Run the save cell (step 11) after any experiment you care about. It is safe to run it more than once.
- **The server is private.** `localhost:8000` exists only inside your Colab machine. Nobody on the internet can reach it. That is what you want here.
- **Numbers vary.** Colab gives you a shared machine, and the exact GPU clock, CPU and disk change between sessions. Expect run-to-run differences of a few percent. The shapes of the curves are what matter.
- **Logs are the truth.** Every server writes its full log to `/content/vllm_logs/<name>.log`. When something fails, the last 40 lines are printed for you. Read them.
""")

# ---------------------------------------------------------------- 1. GPU + install
md("""
## 1. Check the GPU and install vLLM

First, see what Colab gave you. Two facts decide a lot:

- **Memory.** It limits the model size and the KV cache. A T4 has 15 GiB usable, an L4 about 22 GiB, an A100 40 or 80 GiB.
- **Compute capability.** This is NVIDIA's version number for a GPU generation. The T4 is 7.5 (Turing, 2018). Several fast features need 8.0 (Ampere, A100) or 8.9 (Ada, L4) or newer: the bfloat16 number format, FlashAttention kernels and FP8.

If the cell below says "no GPU", go back to **Runtime > Change runtime type**.
""")

code(r'''
import subprocess, json, os, sys, platform

def sh(cmd):
    """Run a shell command and return its stdout as text."""
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

q = sh("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits")
if not q:
    raise SystemExit("No GPU found. Open Runtime > Change runtime type and pick a GPU.")
name, mem_mib, driver, cap = [x.strip() for x in q.splitlines()[0].split(",")]
GPU = {
    "name": name,
    "memory_gib": round(int(mem_mib) / 1024, 1),
    "driver": driver,
    "compute_capability": float(cap),
    "cuda_driver_version": sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | cut -d' ' -f3"),
    "cpu": sh("lscpu | grep 'Model name' | cut -d: -f2").strip(),
    "cpu_count": os.cpu_count(),
    "ram_gib": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
    "python": platform.python_version(),
}
print(json.dumps(GPU, indent=2))
print()
print(sh("nvidia-smi"))
''')

md(f"""
Now install vLLM. We pin version **{VLLM_VERSION}** so your run matches the article. This takes 2 to 5 minutes, mostly downloading. vLLM brings its own PyTorch build, so pip replaces the one Colab ships.

You may see red warnings about version conflicts with other Colab packages (for example `torchaudio` or `fastai`). They do not matter here: vLLM runs in its own process, and we do not use those packages.
""")

code(rf'''
%%time
# uv is preinstalled on Colab and is much faster than pip. Fall back to pip if it is missing.
!uv pip install --system -q "vllm=={VLLM_VERSION}" || pip install -q "vllm=={VLLM_VERSION}"
''')

code(r'''
VLLM_VERSION = sh("vllm --version 2>/dev/null | tail -n 1")
print("vLLM version:", VLLM_VERSION)
assert VLLM_VERSION, "vllm is not on PATH. Re-run the install cell and read its errors."
print("PyTorch used by vLLM:", sh("python -c 'import torch; print(torch.__version__, torch.version.cuda)'"))
''')

# ---------------------------------------------------------------- settings
md("""
## Settings

Change these only if you know why. The defaults are tuned for a free T4.

- `MODEL`: **Qwen2.5-1.5B-Instruct** is small enough to leave a large KV cache on a T4, and good enough to answer questions sensibly. It needs no Hugging Face login. On an L4 or A100 you can try `Qwen/Qwen2.5-7B-Instruct`.
- `MAX_MODEL_LEN`: the longest prompt plus answer one request may have, in tokens. The model supports 32,768, but a smaller limit keeps the experiments fast and the numbers easy to read.
- `GPU_MEMORY_UTILIZATION`: the share of GPU memory vLLM may take. The weights, the working memory and the KV cache all come out of it. The vLLM default is 0.92. We use 0.85 to leave room for Colab's own use of the GPU.
- `QUICK`: cuts every benchmark to about a third of its requests. The numbers get noisier.
""")

code(r'''
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.85
PORT = 8000
API_KEY = "colab-demo-key"          # any string; clients must send it
QUICK = False

# The T4 has no bfloat16 hardware, so we ask for float16 there. Newer GPUs use the model's own type.
DTYPE = "half" if GPU["compute_capability"] < 8.0 else "auto"
SCALE = 0.35 if QUICK else 1.0

LOG_DIR, BENCH_DIR = "/content/vllm_logs", "/content/vllm_bench"
os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(BENCH_DIR, exist_ok=True)

# Everything we measure goes into RESULTS. Step 11 saves it.
RESULTS = {"gpu": GPU, "vllm_version": VLLM_VERSION, "model": MODEL, "dtype": DTYPE,
           "max_model_len": MAX_MODEL_LEN, "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
           "quick": QUICK, "starts": [], "bench": {}}
print(f"model={MODEL} dtype={DTYPE} max_model_len={MAX_MODEL_LEN} quick={QUICK}")
''')

# ---------------------------------------------------------------- helpers
md("""
## Helpers

This cell defines the tools the rest of the notebook uses. Read it once: it is short, and it is the same thing an operator does by hand.

- `start_server(name, extra_args)` runs `vllm serve` in the background, writes its log to a file, waits until `GET /health` answers 200, and reads the important lines out of the startup log.
- `stop_server()` stops it and waits until the GPU memory is free again.
- `metrics()` reads the server's Prometheus page at `/metrics` and returns the numbers as a dictionary.
- `bench(label, ...)` runs `vllm bench serve`, the load generator that ships with vLLM, and saves its JSON result.
""")

code(r'''
import re, time, signal, threading, requests

SERVER = None
SEED = [0]          # every benchmark gets new random prompts, so one run never hits another run's cache
BASE = f"http://localhost:{PORT}"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

LOG_PATTERNS = {
    # name: (regex, how many groups to keep)
    "attention_backend": r"Using (\S+) attention backend",
    "weights_gib_and_load_s": r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds",
    "torch_compile_s": r"torch\.compile took ([\d.]+) s in total",
    "kv_cache_memory_gib": r"Available KV cache memory: ([\d.]+) GiB",
    "kv_cache_tokens": r"KV cache size: ([\d,]+) tokens",
    "max_concurrency": r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x",
    "cuda_graphs_s_and_gib": r"Graph capturing finished in ([\d.]+) secs, took ([\d.]+) GiB",
    "init_engine_s": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+)",
}

def tail(path, n=40):
    with open(path, errors="replace") as f:
        return "".join(f.readlines()[-n:])

def parse_log(path):
    text = open(path, errors="replace").read()
    out = {}
    for key, pat in LOG_PATTERNS.items():
        m = re.findall(pat, text)
        if not m:
            continue
        v = m[-1]
        conv = lambda s: float(s.replace(",", ""))
        out[key] = [conv(x) for x in v] if isinstance(v, tuple) else (v if key == "attention_backend" else conv(v))
    return out

def gpu_used_mib():
    return int(sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").splitlines()[0])

def stop_server():
    global SERVER
    if SERVER is None:
        return
    proc = SERVER["proc"]
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)        # the whole process group: API server + engine
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
    SERVER["log_file"].close()
    SERVER = None
    for _ in range(60):                             # wait until the GPU memory is back
        if gpu_used_mib() < 1000:
            break
        time.sleep(1)

def start_server(name, extra_args=(), timeout=1200):
    """Start `vllm serve` in the background and wait until it is healthy."""
    global SERVER
    stop_server()
    log_path = f"{LOG_DIR}/{name}.log"
    cmd = ["vllm", "serve", MODEL, "--port", str(PORT), "--api-key", API_KEY,
           "--dtype", DTYPE, "--max-model-len", str(MAX_MODEL_LEN),
           "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION), *map(str, extra_args)]
    log_file = open(log_path, "w")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    SERVER = {"proc": proc, "log_file": log_file, "name": name}
    print("$", " ".join(cmd))
    while True:
        if proc.poll() is not None:
            print(tail(log_path)); stop_server()
            raise RuntimeError(f"vllm serve exited with code {proc.returncode}. The log tail is above.")
        try:
            if requests.get(f"{BASE}/health", timeout=2).status_code == 200:
                break
        except requests.exceptions.RequestException:
            pass
        if time.time() - t0 > timeout:
            print(tail(log_path)); stop_server()
            raise TimeoutError("The server did not become healthy in time.")
        time.sleep(1)
    info = {"name": name, "args": list(map(str, extra_args)), "ready_s": round(time.time() - t0, 1),
            **parse_log(log_path), "gpu_used_mib": gpu_used_mib()}
    RESULTS["starts"].append(info)
    print(json.dumps(info, indent=2))
    return info

def metrics():
    """Read /metrics. Sums each metric over its labels; skips histogram buckets."""
    out = {}
    try:
        text = requests.get(f"{BASE}/metrics", timeout=5).text
    except requests.exceptions.RequestException:
        return out                                  # server down: no metrics
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith("vllm:") or "_bucket" in line:
            continue
        name_part, value = line.rsplit(" ", 1)
        name = name_part.split("{")[0]
        try:
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            pass
    return out

SUMMARY_KEYS = ["completed", "failed", "duration", "request_throughput", "output_throughput",
                "total_token_throughput", "max_concurrency", "request_rate", "request_goodput"]

def bench(label, input_len, output_len, num_prompts, rate="inf", concurrency=None,
          prefix_len=0, goodput=None, quiet=True):
    """Run `vllm bench serve` against the running server and keep a summary."""
    num_prompts = max(8, int(num_prompts * SCALE))
    SEED[0] += 1
    cmd = ["vllm", "bench", "serve", "--backend", "openai", "--base-url", BASE,
           "--endpoint", "/v1/completions", "--model", MODEL,
           "--dataset-name", "random", "--random-input-len", str(input_len),
           "--random-output-len", str(output_len), "--random-prefix-len", str(prefix_len),
           "--num-prompts", str(num_prompts), "--request-rate", str(rate),
           "--ignore-eos", "--seed", str(SEED[0]),
           "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
           "--save-result", "--result-dir", BENCH_DIR, "--result-filename", f"{label}.json"]
    if concurrency:
        cmd += ["--max-concurrency", str(concurrency)]
    if goodput:
        cmd += ["--goodput", *goodput]
    env = {**os.environ, "OPENAI_API_KEY": API_KEY}
    before = metrics()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    after = metrics()
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise RuntimeError(f"benchmark {label} failed")
    full = json.load(open(f"{BENCH_DIR}/{label}.json"))
    s = {k: full.get(k) for k in SUMMARY_KEYS}
    s.update({k: v for k, v in full.items() if re.match(r"(mean|median|p\d+)_(ttft|tpot|itl|e2el)_ms$", k)})
    s["num_prompts"] = num_prompts
    s["input_len"], s["output_len"], s["prefix_len"] = input_len, output_len, prefix_len
    s["server"] = SERVER["name"] if SERVER else None
    delta = lambda k: after.get(k, 0) - before.get(k, 0)
    s["preemptions"] = delta("vllm:num_preemptions_total")
    q, h = delta("vllm:prefix_cache_queries_total"), delta("vllm:prefix_cache_hits_total")
    s["prefix_cache_hit_rate"] = round(h / q, 4) if q else None
    RESULTS["bench"][label] = s
    if not quiet:
        print(p.stdout[-2500:])
    print(f"{label}: {s['completed']} ok, {s['failed']} failed | "
          f"{s['output_throughput']:.0f} out tok/s | TTFT p50 {s['median_ttft_ms']:.0f} ms, "
          f"p99 {s['p99_ttft_ms']:.0f} ms | TPOT p50 {s['median_tpot_ms']:.1f} ms")
    return s

print("helpers ready")
''')

# ---------------------------------------------------------------- 2. start
md("""
## 2. Start the server and read its startup log

The next cell runs the same command you would run on a production machine:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 4096 --gpu-memory-utilization 0.85
```

The first start is a **cold start**: it downloads about 3 GB of weights from Hugging Face, then prepares the GPU. Expect 2 to 5 minutes on a T4. While you wait, here is what the server does, in order:

1. **Load the weights** from disk into GPU memory.
2. **Compile the model** with `torch.compile` into faster fused kernels. The result is cached on disk, so the next start is faster.
3. **Profile memory.** It runs one large dummy batch to see the peak working memory.
4. **Make the KV cache.** Whatever is left of the memory budget becomes the KV cache. This decides how many tokens the server can keep in flight.
5. **Capture CUDA graphs.** It records the GPU work of a decode step for many batch sizes, so later steps launch with one call instead of hundreds.
6. **Open the HTTP port.** `GET /health` starts to answer 200.

The cell prints what it found in the log. The key lines are `attention_backend`, `kv_cache_tokens` and `max_concurrency`.
""")

code(r'''
first = start_server("default")
''')

md("""
### Read the result

- `ready_s`: seconds from the command to the first healthy answer. In production this is your **cold start**, and it decides how fast you can add capacity when traffic jumps.
- `weights_gib_and_load_s`: weight memory in GiB and the load time.
- `attention_backend`: the attention kernel vLLM chose. On a T4 it is **`TRITON_ATTN`**, because the FlashAttention and FlashInfer kernels need a newer GPU. On an L4 or A100 you should see `FLASH_ATTN`. This one line explains much of the speed gap between those GPUs.
- `kv_cache_memory_gib` and `kv_cache_tokens`: the memory left for the KV cache, and how many tokens fit in it.
- `max_concurrency`: how many requests of the full `MAX_MODEL_LEN` could have their KV cache in memory at the same time. Real requests are shorter, so more fit.

Look at the raw log too. The cell below prints the lines that matter.
""")

code(r'''
keep = ("non-default args", "attention backend", "Model loading took", "torch.compile took",
        "Available KV cache memory", "KV cache size", "Maximum concurrency", "Graph capturing",
        "init engine", "Starting vLLM API server", "Application startup complete")
for line in open(f"{LOG_DIR}/default.log", errors="replace"):
    if any(k in line for k in keep):
        print(line.rstrip()[:220])
''')

# ---------------------------------------------------------------- 3. first requests
md("""
## 3. The first requests

The server speaks the **OpenAI API**. Any client that can talk to OpenAI can talk to it: change the base URL and the key. That is a big reason vLLM is easy to put into production.

First a normal chat request, then a **streaming** request. With streaming, the server sends each token as soon as it is ready, so we can time two things a user feels:

- **Time to first token (TTFT):** from sending the request to the first token. It covers waiting in the queue plus **prefill**, the step that reads the whole prompt.
- **Time per output token (TPOT):** the average gap between later tokens. Each one is a **decode** step.
""")

code(r'''
from openai import OpenAI
client = OpenAI(base_url=f"{BASE}/v1", api_key=API_KEY)

r = client.chat.completions.create(
    model=MODEL, temperature=0, max_tokens=120,
    messages=[{"role": "user", "content": "Explain in three sentences why an LLM server batches requests."}])
print(r.choices[0].message.content)
print("\nusage:", r.usage)
RESULTS["first_answer"] = r.choices[0].message.content
''')

code(r'''
def stream_once(prompt, max_tokens=256):
    t0 = time.perf_counter(); times = []
    stream = client.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=max_tokens, stream=True,
        stream_options={"include_usage": True},
        messages=[{"role": "user", "content": prompt}])
    usage = None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            times.append(time.perf_counter())
    ttft = (times[0] - t0) * 1000
    n = usage.completion_tokens
    tpot = (times[-1] - times[0]) * 1000 / max(n - 1, 1)
    return {"ttft_ms": round(ttft, 1), "tpot_ms": round(tpot, 2), "output_tokens": n,
            "prompt_tokens": usage.prompt_tokens, "tokens_per_s": round(1000 / tpot, 1)}

stream_once("Say hi.", 8)    # warm-up: the first request after start pays one-time costs
single = [stream_once("Write a short story about a lighthouse keeper.", 256) for _ in range(3)]
RESULTS["single_user_stream"] = single
for s in single:
    print(s)
''')

md("""
One user gets a fast first token and a steady stream. Note the `tokens_per_s` value: that is the speed of **one** conversation. Keep it in mind for step 5, where many users share the GPU.

The server's own view is on its metrics page. These are the numbers a production dashboard plots:
""")

code(r'''
m = metrics()
for k in ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
          "vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:num_preemptions_total",
          "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"]:
    print(f"{k:40s} {m.get(k)}")
''')

# ---------------------------------------------------------------- 4. KV maths
md("""
## 4. Check the KV cache size by hand

The startup log said how many tokens fit in the KV cache. We can predict that number from the model's shape.

For every token, every layer keeps one **key** vector and one **value** vector for each KV head. So the bytes per token are:

$$\\text{bytes per token} = 2 \\times L \\times H_{kv} \\times d_{head} \\times b$$

Here $L$ is the number of layers, $H_{kv}$ the number of key/value heads, $d_{head}$ the size of one head, $b$ the bytes per number (2 for float16), and the 2 in front counts the key and the value.
""")

code(r'''
from huggingface_hub import hf_hub_download
cfg = json.load(open(hf_hub_download(MODEL, "config.json")))
L, H, Hkv = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
d_head = cfg.get("head_dim") or cfg["hidden_size"] // H
bytes_per_token = 2 * L * Hkv * d_head * 2
kv_gib = first["kv_cache_memory_gib"]
predicted_tokens = kv_gib * 2**30 / bytes_per_token
KV = {"layers": L, "heads": H, "kv_heads": Hkv, "head_dim": d_head,
      "bytes_per_token": bytes_per_token, "kv_cache_memory_gib": kv_gib,
      "predicted_tokens": int(predicted_tokens), "log_tokens": int(first["kv_cache_tokens"]),
      "tokens_per_full_request": MAX_MODEL_LEN,
      "full_requests_that_fit": round(first["kv_cache_tokens"] / MAX_MODEL_LEN, 2)}
RESULTS["kv_math"] = KV
print(json.dumps(KV, indent=2))
print(f"\n{bytes_per_token:,} bytes = {bytes_per_token/1024:.0f} KiB per token.")
print(f"Prediction {predicted_tokens:,.0f} tokens vs the log's {first['kv_cache_tokens']:,.0f} tokens.")
''')

md("""
The prediction and the log should agree within about 1%. The small gap comes from rounding the cache to whole **blocks**: vLLM stores the cache in fixed-size pages of tokens, so a partial block at the end is not used.

Qwen2.5-1.5B uses **grouped-query attention**: it has 12 query heads but only 2 KV heads. That makes each token's cache 6 times smaller than with one KV head per query head, and so 6 times more tokens fit.
""")

# ---------------------------------------------------------------- 5. batching
md("""
## 5. Batching: many users at the same time

This is the most important experiment. We send the same kind of request (512 prompt tokens, 128 output tokens) with 1, 2, 4, ... 128 requests **in flight at the same time**. This is a **closed loop**: a new request starts only when one finishes, so concurrency stays fixed.

`--ignore-eos` makes every request write exactly 128 tokens, so all runs do the same work. `random` prompts are random token IDs: their text is noise, but their cost is real.

Watch two columns as concurrency rises:
- **Output throughput** (tokens per second for the whole server) should climb a lot.
- **TPOT** (milliseconds per token for each user) should climb only a little at first.
""")

code(r'''
CONCURRENCY = [1, 2, 4, 8, 16, 32, 64, 128]
for c in CONCURRENCY:
    bench(f"conc_{c}", input_len=512, output_len=128, num_prompts=max(24, 4 * c), concurrency=c)
''')

code(r'''
import pandas as pd, matplotlib.pyplot as plt

rows = [{"concurrency": c, **{k: RESULTS["bench"][f"conc_{c}"][k] for k in
        ["output_throughput", "request_throughput", "median_tpot_ms", "median_ttft_ms", "p99_ttft_ms"]}}
        for c in CONCURRENCY]
df = pd.DataFrame(rows).round(1)
df["per_user_tok_s"] = (1000 / df["median_tpot_ms"]).round(1)
df["speedup_vs_1"] = (df["output_throughput"] / df["output_throughput"][0]).round(2)
display(df)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(df.concurrency, df.output_throughput, "o-"); ax[0].set_xscale("log", base=2)
ax[0].set(xlabel="requests in flight", ylabel="output tokens / s (whole server)", title="Throughput")
ax[1].plot(df.concurrency, df.median_tpot_ms, "o-", color="tab:orange"); ax[1].set_xscale("log", base=2)
ax[1].set(xlabel="requests in flight", ylabel="ms per output token (each user)", title="Per-user speed")
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.show()
''')

md("""
### Read the result

Compare the first and last rows. Throughput grew many times. The time per token for each user grew far less.

That is decode's nature. A decode step must read all the model's weights from GPU memory, and that read takes most of the step time. One request or 64 requests: the weights are read once per step either way. So 64 requests in one step cost little more than one.

The curve flattens when the GPU's arithmetic units, not memory reads, become the limit, or when the batch reaches the server's limits on sequences or tokens per step. **TTFT** also rises at high concurrency: new requests wait for room, and each prefill shares the step with many decodes.
""")

# ---------------------------------------------------------------- 6. load sweep
md("""
## 6. Past capacity: where latency breaks

Real users do not wait for each other. They arrive at random times. This is an **open loop**: requests arrive at a fixed average rate (a Poisson process), whether the server keeps up or not.

We take the best request rate from step 5 as the server's capacity, then send traffic at 25% to 120% of it. Below capacity, latency stays flat. Near and above it, requests queue, and **TTFT** grows without limit.

We also measure **goodput**: the requests per second that meet a latency target, or **SLO** (service level objective). Here the target is TTFT under 1 second and TPOT under 100 ms. These are our choices for a chat product; pick your own for your product.

During the heaviest run, a background thread reads `/metrics` every half second, so we can watch the queue form.
""")

code(r'''
capacity = max(RESULTS["bench"][f"conc_{c}"]["request_throughput"] for c in CONCURRENCY)
RESULTS["capacity_req_s"] = capacity
print(f"capacity from step 5: {capacity:.2f} requests/s")

FRACTIONS = [0.25, 0.5, 0.75, 0.9, 1.0, 1.2]
timeline = []
def sampler(stop_event, t0):
    while not stop_event.is_set():
        m = metrics()
        timeline.append({"t": round(time.time() - t0, 2),
                         "running": m.get("vllm:num_requests_running", 0),
                         "waiting": m.get("vllm:num_requests_waiting", 0),
                         "kv_usage": m.get("vllm:kv_cache_usage_perc", 0)})
        time.sleep(0.5)

for f in FRACTIONS:
    rate = round(f * capacity, 2)
    n = int(min(600, max(60, rate * 60)))          # about one minute of traffic per rate
    if f == FRACTIONS[-1]:
        stop = threading.Event(); th = threading.Thread(target=sampler, args=(stop, time.time())); th.start()
    bench(f"rate_{f}", input_len=512, output_len=128, num_prompts=n, rate=rate,
          goodput=["ttft:1000", "tpot:100"])
    if f == FRACTIONS[-1]:
        stop.set(); th.join()
RESULTS["overload_timeline"] = timeline
''')

code(r'''
rows = []
for f in FRACTIONS:
    b = RESULTS["bench"][f"rate_{f}"]
    rows.append({"load": f, "rate_req_s": b["request_rate"], "achieved_req_s": b["request_throughput"],
                 "goodput_req_s": b.get("request_goodput"), "ttft_p50_ms": b["median_ttft_ms"],
                 "ttft_p99_ms": b["p99_ttft_ms"], "tpot_p50_ms": b["median_tpot_ms"], "tpot_p99_ms": b["p99_tpot_ms"]})
dr = pd.DataFrame(rows).round(2); display(dr)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(dr.load, dr.ttft_p50_ms, "o-", label="TTFT p50"); ax[0].plot(dr.load, dr.ttft_p99_ms, "o-", label="TTFT p99")
ax[0].set(xlabel="offered load (fraction of capacity)", ylabel="ms", title="Time to first token", yscale="log"); ax[0].legend()
tl = pd.DataFrame(timeline)
if len(tl):
    ax[1].plot(tl.t, tl.running, label="running"); ax[1].plot(tl.t, tl.waiting, label="waiting")
    ax2 = ax[1].twinx(); ax2.plot(tl.t, tl.kv_usage * 100, "k:", label="KV cache %"); ax2.set_ylabel("KV cache used, %")
    ax[1].set(xlabel="seconds", ylabel="requests", title=f"Inside the server at {FRACTIONS[-1]:.0%} load"); ax[1].legend(loc="upper left")
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.show()
''')

md("""
### Read the result

- Up to about 75% load, TTFT barely moves. The server has spare room in each step.
- Near 100%, p99 TTFT starts to climb first. A short burst of arrivals has to wait.
- Above 100%, arrivals beat completions. The `waiting` line in the right chart grows for as long as the overload lasts, and TTFT grows with it. Throughput does not rise any more; only waiting does.
- **Goodput** falls below the arrival rate once requests miss the SLO. This is the number to plan capacity with, not the raw throughput.

In production you keep the server below the knee, often at 60% to 80% of measured capacity, and you add replicas before the queue forms.
""")

# ---------------------------------------------------------------- 7. prefix caching
md("""
## 7. Prefix caching: reuse the shared prompt

Many products send the same long start with every request: a system prompt, tool definitions, a document. vLLM can keep the KV cache of that shared start and reuse it, so later requests skip most of the prefill. This is **prefix caching**, and it is on by default.

The workload: every request has the same 2,048-token prefix, then 128 random tokens, and asks for 64 output tokens. We run it on the current server (caching on), then restart with `--no-enable-prefix-caching` and run it again.

The restart also shows a **warm start**: the weights are already on disk, and the compiled graph is cached.
""")

code(r'''
PREFIX_JOB = dict(input_len=128, output_len=64, prefix_len=2048, num_prompts=160, concurrency=8)
bench("prefix_on", **PREFIX_JOB)
start_server("no_prefix_cache", ["--no-enable-prefix-caching"])
bench("prefix_off", **PREFIX_JOB)
''')

code(r'''
on, off = RESULTS["bench"]["prefix_on"], RESULTS["bench"]["prefix_off"]
dp = pd.DataFrame([{"prefix cache": n, "TTFT p50 ms": b["median_ttft_ms"], "TTFT p99 ms": b["p99_ttft_ms"],
                    "req/s": b["request_throughput"], "hit rate": b["prefix_cache_hit_rate"]}
                   for n, b in [("on", on), ("off", off)]]).round(3)
display(dp)
print(f"TTFT p50 is {off['median_ttft_ms'] / on['median_ttft_ms']:.1f}x lower with the cache on.")
cold, warm = RESULTS["starts"][0], RESULTS["starts"][-1]
print(f"Startup: cold {cold['ready_s']} s, warm {warm['ready_s']} s.")
''')

md("""
With the cache on, the hit rate is high: almost every prefix token is found in the cache. Each request then prefills only its own 128 tokens instead of about 2,176. TTFT drops by several times, and throughput rises because the GPU does less work per request.

The warm start is faster than the cold start because the download and part of the compile are skipped. In production, keeping weights on a fast local disk and keeping the compile cache are the cheapest ways to cut cold starts.
""")

# ---------------------------------------------------------------- 8. preemption
md("""
## 8. A small KV cache: preemption

What happens when the KV cache is too small for the load? vLLM starts requests while there is room for their prompts. As they write their answers, their caches grow. When the cache is full, the scheduler **preempts** a request: it frees that request's blocks and later computes them again from the start. The request is not lost, but its work is done twice.

We restart the server with a KV cache of only 6,144 tokens (`--kv-cache-memory-bytes`), then send 32 requests at once with 512 prompt tokens and 512 output tokens each. Together they need 32,768 tokens of cache. We run the same job on a normal server first, for comparison.
""")

code(r'''
PRE_JOB = dict(input_len=512, output_len=512, num_prompts=96, concurrency=32)
start_server("default_2")
bench("preempt_baseline", **PRE_JOB)
small_bytes = int(KV["bytes_per_token"] * 6144)
start_server("small_kv", ["--kv-cache-memory-bytes", small_bytes])
bench("preempt_small_kv", **PRE_JOB)
''')

code(r'''
base, small = RESULTS["bench"]["preempt_baseline"], RESULTS["bench"]["preempt_small_kv"]
dq = pd.DataFrame([{"server": n, "KV tokens": s["kv_cache_tokens"], "preemptions": b["preemptions"],
                    "out tok/s": b["output_throughput"], "TTFT p50 ms": b["median_ttft_ms"],
                    "TTFT p99 ms": b["p99_ttft_ms"], "E2E p99 ms": b["p99_e2el_ms"]}
                   for n, b, s in [("normal", base, RESULTS["starts"][-2]), ("small KV", small, RESULTS["starts"][-1])]]).round(1)
display(dq)
''')

md("""
The normal server should show zero preemptions. The small one runs fewer requests at a time, preempts some of them, and redoes their work. Throughput falls and the tail latency grows.

In production, `vllm:num_preemptions_total` rising is an alarm: the KV cache is too small for the traffic. The fixes are more memory for the cache (a higher `--gpu-memory-utilization`, a smaller model or quantized weights), a lower `--max-num-seqs`, or more replicas.
""")

# ---------------------------------------------------------------- 9. failures
md("""
## 9. Break it on purpose

Production skill is knowing what failures look like before they happen. We restart a normal server and try three.
""")

code(r'''
start_server("default_3")
FAIL = {}

# (a) A prompt longer than MAX_MODEL_LEN
r = requests.post(f"{BASE}/v1/completions", headers=HEADERS,
                  json={"model": MODEL, "prompt": "hello " * (MAX_MODEL_LEN + 500), "max_tokens": 16})
FAIL["too_long"] = {"status": r.status_code, "body": r.text[:400]}

# (b) A wrong API key
r = requests.post(f"{BASE}/v1/completions", headers={"Authorization": "Bearer wrong"},
                  json={"model": MODEL, "prompt": "hi", "max_tokens": 4})
FAIL["wrong_key"] = {"status": r.status_code, "body": r.text[:200]}

# (c) max_tokens that does not fit: prompt plus answer longer than MAX_MODEL_LEN
r = requests.post(f"{BASE}/v1/completions", headers=HEADERS,
                  json={"model": MODEL, "prompt": "hi", "max_tokens": MAX_MODEL_LEN + 10})
FAIL["max_tokens_too_big"] = {"status": r.status_code, "body": r.text[:400]}

for k, v in FAIL.items():
    print(k, v["status"], "\n   ", v["body"][:300], "\n")
''')

md("""
These are clean failures: the server rejects the request at once with a clear status code (400 for a bad request, 401 for a bad key) and keeps serving everyone else. Your client should treat 4xx as "do not retry".

Now the ugly one: the server dies while it is under load. We start a benchmark in the background and kill the server after 15 seconds.
""")

code(r'''
def run_bench_bg(out):
    try:
        out["result"] = bench("killed_mid_run", input_len=512, output_len=256, num_prompts=200, concurrency=32)
    except Exception as e:
        out["error"] = repr(e)[:300]

out = {}
th = threading.Thread(target=run_bench_bg, args=(out,)); th.start()
time.sleep(15)
print("killing the server now")
os.killpg(SERVER["proc"].pid, signal.SIGKILL)
th.join(timeout=300)
try:
    health = requests.get(f"{BASE}/health", timeout=2).status_code
except requests.exceptions.RequestException as e:
    health = type(e).__name__
FAIL["killed_mid_run"] = {"health_after": health, **({"completed": out["result"]["completed"], "failed": out["result"]["failed"]}
                           if "result" in out else {"bench_error": out.get("error")})}
RESULTS["failures"] = FAIL
print(json.dumps(FAIL["killed_mid_run"], indent=2))
SERVER["log_file"].close(); SERVER = None
''')

md("""
Every request in flight is lost, and `/health` stops answering. Nothing inside vLLM can save them: the KV cache lived in that process's GPU memory.

This is why production setups run **several replicas behind a load balancer** that checks `/health`, and why clients **retry with backoff** on connection errors and 5xx answers (but never on 4xx). A retry sent to a healthy replica turns a crash into a short delay.
""")

# ---------------------------------------------------------------- 10. FP8
md("""
## 10. (L4 or A100 only) FP8 weights

**FP8** stores each weight in 8 bits instead of 16. vLLM can convert the weights while it loads them (`--quantization fp8`). The weights take half the memory, so the KV cache grows, and decode reads half the bytes per step.

FP8 matrix hardware needs compute capability 8.9 (L4, H100). On the T4 this cell skips itself. On an A100 (8.0), vLLM runs FP8 weights with a slower path; the memory saving is real, the speed gain may not be.
""")

code(r'''
if GPU["compute_capability"] < 8.0:
    print("Skipped: FP8 needs a newer GPU than", GPU["name"])
else:
    start_server("fp8", ["--quantization", "fp8"])
    bench("fp8_conc_64", input_len=512, output_len=128, num_prompts=256, concurrency=64)
    start_server("bf16_again")
    bench("bf16_conc_64", input_len=512, output_len=128, num_prompts=256, concurrency=64)
    fp8s, bfs = [s for s in RESULTS["starts"] if s["name"] == "fp8"][-1], RESULTS["starts"][-1]
    display(pd.DataFrame([
        {"weights": "16-bit", "weights GiB": bfs["weights_gib_and_load_s"][0], "KV tokens": bfs["kv_cache_tokens"],
         "out tok/s": RESULTS["bench"]["bf16_conc_64"]["output_throughput"]},
        {"weights": "FP8", "weights GiB": fp8s["weights_gib_and_load_s"][0], "KV tokens": fp8s["kv_cache_tokens"],
         "out tok/s": RESULTS["bench"]["fp8_conc_64"]["output_throughput"]}]).round(1))
    stop_server()
''')

# ---------------------------------------------------------------- 11. save
md("""
## 11. Save everything

This writes `vllm_colab_results.json` (every number above, with the GPU and versions) and `vllm_colab_run.zip` (the results, every server log and every raw benchmark file), then downloads both. You can run this cell at any time; it saves what exists so far.

If the download does not start, open the folder icon on the left and download the files from `/content`.
""")

code(r'''
import shutil, datetime
stop_server()
RESULTS["saved_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
json.dump(RESULTS, open("/content/vllm_colab_results.json", "w"), indent=2, default=str)
os.makedirs("/content/vllm_run", exist_ok=True)
for d in (LOG_DIR, BENCH_DIR):
    shutil.copytree(d, f"/content/vllm_run/{os.path.basename(d)}", dirs_exist_ok=True)
shutil.copy("/content/vllm_colab_results.json", "/content/vllm_run/")
shutil.make_archive("/content/vllm_colab_run", "zip", "/content/vllm_run")
print("saved /content/vllm_colab_results.json and /content/vllm_colab_run.zip")
try:
    from google.colab import files
    files.download("/content/vllm_colab_results.json"); files.download("/content/vllm_colab_run.zip")
except Exception as e:
    print("Download it from the file panel:", e)
''')

# ---------------------------------------------------------------- notes
md("""
## Troubleshooting

| What you see | Why | What to do |
|---|---|---|
| `Bfloat16 is only supported on GPUs with compute capability of at least 8.0` | You changed `DTYPE` on a T4 | Keep `DTYPE = "half"` on a T4 |
| `CUDA out of memory` at startup | Another process holds GPU memory, often a server from an earlier run | Run `stop_server()`, or **Runtime > Restart session**, then run from the settings cell again |
| `The model's max seq len ... is larger than the maximum number of tokens that can be stored in KV cache` | `MAX_MODEL_LEN` too large for the memory left | Lower `MAX_MODEL_LEN`, raise `GPU_MEMORY_UTILIZATION`, or use a smaller model |
| The server never becomes healthy | Usually a slow download or a crash; the log tail is printed | Read the log in `/content/vllm_logs/` |
| `Address already in use` | An old server still holds port 8000 | `!pkill -f "vllm serve"`, wait 10 s, start again |
| Runtime disconnected | Colab idle or session limit | Reconnect and run from the top; the pip install is faster the second time only within the same session |

## What this notebook cannot show

Colab is a good place to learn the engine. It is not a production environment. These parts need a rented GPU machine or a cluster:

- **Docker.** Production runs the official `vllm/vllm-openai` image; Colab cannot run containers.
- **Dashboards and alerts.** Prometheus and Grafana scraping `/metrics` over time.
- **More than one GPU.** Tensor parallelism (`--tensor-parallel-size`) needs several GPUs in one machine.
- **Several replicas and a router.** Load balancing, prefix-aware routing, autoscaling and rolling updates, for example with the vLLM production stack or llm-d on Kubernetes.
- **Real traffic shapes.** Real prompts, real output lengths and real bursts. Random tokens with fixed lengths make clean experiments, but your traffic is different.

The article explains each result in depth and links to the rest of the series: [LLM Inference, Part 8]({ARTICLE_URL}).
""".replace("{ARTICLE_URL}", ARTICLE_URL))


# ---------------------------------------------------------------- write
def to_lines(src):
    lines = src.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"name": "vllm-on-colab.ipynb", "provenance": [], "toc_visible": True,
                  "gpuType": "T4", "include_colab_link": False},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "title": "Run vLLM Like Production, on a Free Colab GPU",
        "authors": [{"name": "Ishwar Jangid"}],
        "vllm_version": VLLM_VERSION,
    },
    "cells": [],
}
for i, (kind, src) in enumerate(cells):
    cell = {"cell_type": kind, "metadata": {"id": f"cell-{i:02d}"}, "source": to_lines(src)}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    nb["cells"].append(cell)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}: {len(cells)} cells")
