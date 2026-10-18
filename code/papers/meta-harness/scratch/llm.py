"""The frozen task model M of the paper (§3): a small instruct model that the harness calls with a prompt.

The harness never changes the model's weights. It only decides what text goes into the prompt.
Default: Qwen/Qwen2.5-0.5B-Instruct, greedy decoding, so the same prompt always gives the same answer.

Two ways to generate, so a reader can see what model.generate() does:
  greedy_from_scratch()  our own loop: run the model, take the most likely next token, append it, repeat
  LocalLLM.__call__()    the library way: transformers' model.generate(do_sample=False)
llm_check.py (one folder up: mh_part6_llm.py) shows that both give the same text."""
import os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TASK_MODEL = os.environ.get('MH_TASK_MODEL', 'Qwen/Qwen2.5-0.5B-Instruct')


def pick_device():
    if os.environ.get('MH_DEVICE'):
        return os.environ['MH_DEVICE']
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


class LocalLLM:
    """A chat model behind one call: llm(prompt) -> reply text. It also counts calls and prompt characters."""

    def __init__(self, name=TASK_MODEL, device=None, dtype=torch.float16):
        self.name, self.device = name, device or pick_device()
        if self.device == 'cpu':
            dtype = torch.float32
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(self.device).eval()
        self.calls, self.prompt_chars, self.seconds = 0, 0, 0.0

    def chat_ids(self, prompt, system=None):
        """Wrap the prompt in the model's chat template and return the token ids, shape (1, n)."""
        msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors='pt').input_ids.to(self.device)

    @torch.no_grad()
    def __call__(self, prompt, max_new_tokens=16, system=None, temperature=0.0, seed=None):
        t0 = time.time()
        ids = self.chat_ids(prompt, system)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.eos_token_id)
        if temperature > 0:
            if seed is not None:
                torch.manual_seed(seed)
            kw.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        out = self.model.generate(ids, attention_mask=torch.ones_like(ids), **kw)
        reply = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        self.calls += 1
        self.prompt_chars += len(prompt)
        self.seconds += time.time() - t0
        return reply


@torch.no_grad()
def greedy_from_scratch(llm, prompt, max_new_tokens=16):
    """Method 1: greedy decoding written out. No KV cache, so it is slower, but every step is visible."""
    ids = llm.chat_ids(prompt)                                   # (1, n) the prompt tokens
    start = ids.shape[1]
    for _ in range(max_new_tokens):
        logits = llm.model(ids).logits                           # (1, n, vocab): a score for every next token
        next_id = logits[0, -1].argmax()                         # the most likely token after the last position
        if next_id.item() == llm.tok.eos_token_id or next_id.item() in llm.tok.all_special_ids:
            break                                                # the model says "I am done"
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)        # (1, n + 1): append and go again
    return llm.tok.decode(ids[0, start:], skip_special_tokens=True).strip()
