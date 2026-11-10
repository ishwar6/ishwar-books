"""Inside a DeepSeek MoE, live: run DeepSeek-V2-Lite-Chat on this machine and stream every routing decision.

    ~/.venvs/attn/bin/python code/deepseek/moe-viewer/server.py      # then open http://localhost:8765

What is real: the model, the tokens, the 6 experts each MoE layer picks for each token (captured with forward hooks
on the routers), the router weights, and the time of every step on this machine.
What is simulated (in the page, not here): where each expert would live on a machine with a small GPU, some RAM
and an SSD. The page replays the real choices through that cache and counts what would be read from where.

Every event is also printed to the terminal and appended to code/deepseek/moe-viewer/logs/<date>.log.
Env: MODEL (default deepseek-ai/DeepSeek-V2-Lite-Chat), PORT (8765), DEVICE (mps / cuda / cpu, auto),
     EXPERTS (batched_mm, eager or grouped_mm: how transformers runs the experts).
"""
import json, os, sys, threading, time, datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

HERE = Path(__file__).resolve().parent
MODEL = os.environ.get('MODEL', 'deepseek-ai/DeepSeek-V2-Lite-Chat')
PORT = int(os.environ.get('PORT', 8765))
DEVICE = os.environ.get('DEVICE') or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
LOG_DIR = HERE / 'logs'; LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = open(LOG_DIR / f'{datetime.date.today()}.log', 'a', buffering=1)


def log(msg):
    line = f'{datetime.datetime.now():%H:%M:%S.%f}'[:-3] + '  ' + msg
    print(line, flush=True)
    LOG_FILE.write(line + '\n')
    return line


# ---------------------------------------------------------------- load the model
log(f'loading {MODEL} on {DEVICE} ...')
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
dtype = torch.bfloat16 if DEVICE != 'cpu' else torch.float32
# 'batched_mm' gathers the picked experts' weights and runs one batched matmul. On an Apple GPU it decoded
# OLMoE at about 240 ms per token against about 2,000 ms for the default loop over experts (measured 2026-11-10).
EXPERTS_IMPL = os.environ.get('EXPERTS', 'batched_mm')
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, experts_implementation=EXPERTS_IMPL).to(DEVICE).eval()
cfg = model.config
LOAD_S = time.time() - t0

# Find the routers: one per MoE layer. Each returns (logits, topk_weights, topk_indices).
ROUTERS = [(name, m) for name, m in model.named_modules() if type(m).__name__.lower().endswith('topkrouter')]
assert ROUTERS, 'no router modules found: this viewer expects a DeepSeek-style MoE'
MOE_LAYERS = [int(name.split('layers.')[1].split('.')[0]) for name, _ in ROUTERS]
CAPTURE = [None] * len(ROUTERS)


def make_hook(i):
    def hook(_module, _inp, out):
        _logits, w, idx = out
        CAPTURE[i] = (idx.detach().to('cpu'), w.detach().float().to('cpu'))
    return hook


for i, (_, m) in enumerate(ROUTERS):
    m.register_forward_hook(make_hook(i))

H = cfg.hidden_size
I = getattr(cfg, 'moe_intermediate_size', None) or cfg.intermediate_size      # OLMoE has no separate MoE size
N_EXP = getattr(cfg, 'n_routed_experts', None) or cfg.num_experts
N_SHARED = getattr(cfg, 'n_shared_experts', 0) or 0
BYTES = torch.finfo(dtype).bits // 8
EXPERT_BYTES = 3 * H * I * 2                     # gate, up and down projections, counted in bf16 (2 bytes)
n_params = sum(p.numel() for p in model.parameters())
expert_params = 3 * H * I * N_EXP * len(ROUTERS)
INFO = {
    'model': MODEL, 'device': DEVICE, 'experts_impl': EXPERTS_IMPL, 'dtype': str(dtype).replace('torch.', ''), 'load_s': round(LOAD_S, 1),
    'layers': cfg.num_hidden_layers, 'moe_layers': MOE_LAYERS, 'n_experts': N_EXP,
    'top_k': cfg.num_experts_per_tok, 'n_shared': N_SHARED, 'is_deepseek': 'deepseek' in cfg.model_type, 'hidden': H, 'expert_hidden': I,
    'expert_bytes': EXPERT_BYTES, 'params_total': n_params, 'params_routed_experts': expert_params,
    'params_always_used': n_params - expert_params,
    'params_active_per_token': n_params - expert_params + 3 * H * I * cfg.num_experts_per_tok * len(ROUTERS),
}
log(f'loaded in {LOAD_S:.1f} s: {n_params / 1e9:.2f}B params, {len(ROUTERS)} MoE layers x '
    f'{N_EXP} experts (+{N_SHARED} shared), top-{cfg.num_experts_per_tok}, '
    f'expert = {EXPERT_BYTES / 2**20:.1f} MiB')
log(f'active per token: {INFO["params_active_per_token"] / 1e9:.2f}B of {n_params / 1e9:.2f}B')

GEN_LOCK = threading.Lock()


def routing_for_token(t):
    """For token position t of the last forward pass: per MoE layer, the expert ids and weights."""
    ids = [CAPTURE[i][0][t].tolist() for i in range(len(ROUTERS))]
    ws = [[round(x, 4) for x in CAPTURE[i][1][t].tolist()] for i in range(len(ROUTERS))]
    return ids, ws


def generate(prompt, max_new, temperature, emit):
    if tok.chat_template:
        msgs = [{'role': 'user', 'content': prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt', return_dict=True)['input_ids'].to(DEVICE)
    else:                                       # a base model: continue the text as written
        ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    n_prompt = ids.shape[1]
    cache = DynamicCache()
    emit('log', log(f'request: {n_prompt} prompt tokens, max_new={max_new}, temperature={temperature}'))

    # ---- prefill: the whole prompt in one forward pass
    t = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    if DEVICE == 'mps': torch.mps.synchronize()
    pre_ms = (time.perf_counter() - t) * 1000
    prompt_tokens = tok.convert_ids_to_tokens(ids[0].tolist())
    pre_routes = [routing_for_token(j) for j in range(n_prompt)]
    emit('prefill', {'n': n_prompt, 'ms': round(pre_ms, 1),
                     'tokens': [tok.convert_tokens_to_string([p]) for p in prompt_tokens],
                     'routes': [r[0] for r in pre_routes]})
    emit('log', log(f'prefill: {n_prompt} tokens in one pass, {pre_ms:.0f} ms '
                    f'({n_prompt * len(ROUTERS) * cfg.num_experts_per_tok:,} expert picks)'))

    logits = out.logits[:, -1, :]
    produced, t_start = 0, time.perf_counter()
    for step in range(max_new):
        if temperature > 0:
            nxt = torch.multinomial(torch.softmax(logits.float() / temperature, -1), 1)
        else:
            nxt = logits.argmax(-1, keepdim=True)
        tid = nxt.item()
        if tid == tok.eos_token_id:
            emit('log', log('end of answer (eos token)'))
            break
        # ---- decode: one new token through every layer
        t = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=nxt.to(DEVICE), past_key_values=cache, use_cache=True)
        if DEVICE == 'mps': torch.mps.synchronize()
        ms = (time.perf_counter() - t) * 1000
        logits = out.logits[:, -1, :]
        r_ids, r_ws = routing_for_token(0)
        text = tok.decode([tid])
        produced += 1
        emit('token', {'i': step, 'id': tid, 'text': text, 'ms': round(ms, 1), 'routes': r_ids, 'weights': r_ws})
        top = ' | '.join(f'L{MOE_LAYERS[k]}:{",".join(map(str, sorted(r_ids[k])))}' for k in (0, len(r_ids) // 2, len(r_ids) - 1))
        emit('log', log(f'token {step:>3} {text!r:<14} {ms:6.1f} ms   {top}'))
    total = time.perf_counter() - t_start
    stats = {'prompt_tokens': n_prompt, 'new_tokens': produced, 'prefill_ms': round(pre_ms, 1),
             'decode_s': round(total, 2), 'tok_per_s': round(produced / total, 2) if total else 0}
    emit('done', stats)
    emit('log', log(f'done: {produced} tokens in {total:.2f} s = {stats["tok_per_s"]} tokens/s on {DEVICE}'))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            return self._send(200, (HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        if self.path == '/info':
            return self._send(200, json.dumps(INFO), 'application/json')
        self._send(404, 'not found', 'text/plain')

    def do_POST(self):
        if self.path != '/generate':
            return self._send(404, 'not found', 'text/plain')
        req = json.loads(self.rfile.read(int(self.headers['Content-Length'])) or b'{}')
        if not GEN_LOCK.acquire(blocking=False):
            return self._send(409, json.dumps({'error': 'busy: one answer at a time'}), 'application/json')
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream'); self.send_header('Cache-Control', 'no-cache')
            self.end_headers()

            def emit(kind, data):
                self.wfile.write(f'event: {kind}\ndata: {json.dumps(data)}\n\n'.encode()); self.wfile.flush()
            generate(str(req.get('prompt', ''))[:4000], min(int(req.get('max_new', 200)), 1000),
                     float(req.get('temperature', 0)), emit)
        except (BrokenPipeError, ConnectionResetError):
            log('client stopped the answer')
        finally:
            GEN_LOCK.release()


if __name__ == '__main__':
    log(f'open http://localhost:{PORT}')
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
