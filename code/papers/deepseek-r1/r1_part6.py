"""Part 6 (distillation, §2.4 and §4.1): the SFT loss of distillation, and what it does to a model.

1. Generate one real R1-style trace with deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (a GSM8K problem).
2. The SFT loss of distillation, two ways, on that trace and the student Qwen/Qwen2.5-Math-1.5B:
   Method 1 from scratch (chat template, mask the prompt with -100, log-softmax by hand, mean over response tokens);
   Method 2 the library: TRL's SFTTrainer tokenises and masks the same example, then transformers computes the loss
   from `labels`. Both must agree to 1e-5.
3. Teacher text under two models: the per-token loss of R1 text under the base model (before distillation) and the
   distilled model (after). Texts: our generated trace, three real DeepSeek-R1 (671B) traces from the public
   s1K-1.1 dataset, and, as a control, an answer written by the base model itself.
All model work runs inside common.heavy('part6'). Seeds are set. Device: mps (falls back to cpu)."""
import collections, contextlib, json, math, os, re
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from common import Log, save, heavy, RESULTS

DISTILL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
BASE = 'Qwen/Qwen2.5-Math-1.5B'
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'
log = Log('part6_loss')
R = {}


def load(name, dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(DEV).eval()
    return tok, model


# ---------------------------------------------------------------- the SFT example: prompt + teacher response
def build_example(tok, question, response):
    """Tokenise one distillation example the way SFT sees it.
    prompt: the chat template up to where the assistant starts (it ends with '<think>\\n' for the R1 template).
    response: the teacher's text plus the end-of-sequence token. Only the response is trained on."""
    msgs = [{'role': 'user', 'content': question + SUFFIX}]
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)   # list of P ids
    if not isinstance(prompt_ids, list):
        prompt_ids = prompt_ids['input_ids']
    response_ids = tok(response + tok.eos_token, add_special_tokens=False)['input_ids']  # list of N ids
    input_ids = prompt_ids + response_ids                                                 # P + N ids
    labels = [-100] * len(prompt_ids) + response_ids                                      # -100 = "no loss here"
    return input_ids, labels, len(prompt_ids)


def sft_loss_from_scratch(logits, labels):
    """Method 1. logits: (T, V) floats, one row per position. labels: (T,) ids, -100 where there is no loss.
    The model at position t predicts the token at t + 1, so we shift by one."""
    logits = logits[:-1]                                      # (T-1, V): predictions made at positions 0 .. T-2
    target = labels[1:]                                       # (T-1,):   the tokens at positions 1 .. T-1
    keep = target != -100                                     # (T-1,):   True only on response tokens
    m = logits.max(dim=-1, keepdim=True).values               # (T-1, 1): row maximum, for a stable log-sum-exp
    logsumexp = m + torch.log(torch.exp(logits - m).sum(dim=-1, keepdim=True))   # (T-1, 1)
    logprobs = logits - logsumexp                             # (T-1, V): log-softmax by hand
    picked = logprobs[keep].gather(1, target[keep].unsqueeze(1)).squeeze(1)      # (N,): log p(correct token)
    nll = -picked                                             # (N,): loss of each response token
    return nll.mean(), nll                                    # mean over response tokens only


def per_token_nll(model, prompt_ids, response_ids):
    """Loss of each response token (natural log), given the prompt. Returns a (N,) tensor on the cpu."""
    ids = torch.tensor([prompt_ids + response_ids], device=DEV)
    with torch.no_grad():
        logits = model(ids).logits[0].float()                 # (T, V)
    labels = torch.tensor([-100] * len(prompt_ids) + response_ids, device=DEV)
    _, nll = sft_loss_from_scratch(logits, labels)
    return nll.cpu()


def main(lock=True):
    """lock=False when r1_part6_run.py already holds the heavy-job lock."""
    torch.manual_seed(0)
    log('$ python r1_part6.py')
    gsm = load_dataset('openai/gsm8k', 'main', split='test')
    q = gsm[0]['question']
    gold = gsm[0]['answer'].split('####')[-1].strip()

    with (heavy('part6') if lock else contextlib.nullcontext()):
        # ============================================================ 1. one R1-style trace from the distilled model
        log('== 1. one trace from DeepSeek-R1-Distill-Qwen-1.5B (GSM8K test #0, T=0.6, top-p 0.95, max 1024 new tokens)')
        dtok, dmodel = load(DISTILL)
        msgs = [{'role': 'user', 'content': q + SUFFIX}]
        p = dtok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(DEV)
        torch.manual_seed(0)
        out = dmodel.generate(**p, max_new_tokens=1024, do_sample=True, temperature=0.6, top_p=0.95)
        gen = out[0, p['input_ids'].shape[1]:]
        finished = bool((gen == dtok.eos_token_id).any())
        trace = dtok.decode(gen, skip_special_tokens=True)
        log(f'question: {q}')
        log(f'gold answer: {gold}')
        log(f'generated tokens: {len(gen)}, finished with end token: {finished}')
        log('trace starts:', repr(trace[:220]))
        log('trace ends:  ', repr(trace[-160:]))
        R['trace'] = dict(question=q, gold=gold, text=trace, n_tokens=len(gen), finished=finished)

        # ============================================================ 2. the SFT loss, two ways, student = base model
        log('== 2. SFT loss of this example for the student Qwen2.5-Math-1.5B (fp32)')
        btok, bmodel = load(BASE)
        # The student is trained in the R1 chat format, so we use the distilled model's tokenizer and template.
        input_ids, labels, P = build_example(dtok, q, trace)
        log(f'tokens: {len(input_ids)} in total, {P} prompt tokens (label -100), {len(input_ids) - P} response tokens')
        for t in range(P - 3, P + 3):
            log(f'  position {t:3d}  token {dtok.decode([input_ids[t]])!r:22s} label {labels[t]}')
        ids = torch.tensor([input_ids], device=DEV)
        lab = torch.tensor([labels], device=DEV)
        with torch.no_grad():
            logits = bmodel(ids).logits[0].float()             # (T, V) = (len(input_ids), 151936)
        log(f'logits shape: {tuple(logits.shape)}')
        loss1, nll1 = sft_loss_from_scratch(logits, lab[0])
        log(f'Method 1 (from scratch)        loss = {loss1.item():.6f}  over {nll1.numel()} tokens')

        # Method 2a: transformers computes the loss itself when we pass labels
        with torch.no_grad():
            loss2 = bmodel(ids, labels=lab).loss.float()
        log(f'Method 2a (model(labels=...))  loss = {loss2.item():.6f}')
        # Method 2b: PyTorch's cross_entropy with ignore_index=-100
        loss2b = F.cross_entropy(logits[:-1], lab[0, 1:], ignore_index=-100)
        log(f'Method 2b (F.cross_entropy)    loss = {loss2b.item():.6f}')

        # Method 2c: TRL's SFTTrainer builds the example (tokenise, mask the prompt) and computes the loss
        try:
            from trl import SFTConfig, SFTTrainer
            from datasets import Dataset
            prompt_text = dtok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            # this tokenizer does not add the begin token by itself, so the prompt text keeps it
            ds = Dataset.from_list([{'prompt': prompt_text, 'completion': trace + dtok.eos_token}])
            cfg = SFTConfig(output_dir=os.path.join(RESULTS, '.part6_trl_tmp'), report_to=[], completion_only_loss=True,
                            max_length=None, per_device_eval_batch_size=1)
            trainer = SFTTrainer(model=bmodel, args=cfg, train_dataset=ds, processing_class=dtok)
            ex = trainer.train_dataset[0]
            same_ids = list(ex['input_ids']) == input_ids
            same_labels = list(ex['labels']) == labels
            log(f'TRL tokenised example: same input ids as ours: {same_ids}, same labels (prompt masked): {same_labels}')
            batch = trainer.data_collator([ex])
            batch = {k: v.to(DEV) for k, v in batch.items()}
            with torch.no_grad():
                loss3 = trainer.compute_loss(bmodel, batch).float()
            log(f'Method 2c (TRL SFTTrainer.compute_loss) loss = {loss3.item():.6f}')
        except Exception as e:                                 # keep the other results if TRL changes
            log(f'Method 2c failed: {type(e).__name__}: {e}')
            loss3, same_ids, same_labels = loss2, None, None
        d = max(abs(loss1.item() - x.item()) for x in (loss2, loss2b, loss3))
        log(f'largest difference between methods: {d:.2e}  (agree to 1e-5: {d < 1e-5})')
        R['loss'] = dict(P=P, N=len(input_ids) - P, m1=loss1.item(), m2a=loss2.item(), m2b=loss2b.item(),
                         m2c=loss3.item(), diff=d, same_ids=same_ids, same_labels=same_labels)
        # a few per-token losses to read
        toks = [dtok.decode([i]) for i in input_ids[P:]]
        log('first 12 response tokens and their loss under the student (nats):')
        log('  ' + '  '.join(f'{t!r}:{v:.2f}' for t, v in zip(toks[:12], nll1[:12].tolist())))

        # ============================================================ 3. teacher text under base vs distilled
        log('== 3. per-token loss of R1 text under the base model and the distilled model (fp32)')
        # real DeepSeek-R1 traces: the three shortest correct ones among the first 300 rows of s1K-1.1
        rows = []
        for i, r in enumerate(load_dataset('simplescaling/s1K-1.1', split='train', streaming=True)):
            if i >= 300:
                break
            if r['deepseek_grade'] == 'Yes':
                rows.append(r)
        rows.sort(key=lambda r: len(r['deepseek_thinking_trajectory']))
        real = rows[:3]
        # the control: an answer written by the base model itself, in its own chat format
        bmsgs = [{'role': 'user', 'content': q}]
        bp = btok.apply_chat_template(bmsgs, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(DEV)
        torch.manual_seed(0)
        bout = bmodel.generate(**bp, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.95)
        base_text = btok.decode(bout[0, bp['input_ids'].shape[1]:], skip_special_tokens=True)
        log('base model answer to the same question starts:', repr(base_text[:160]))

        # Only the thinking part: '</think>' is one special token for the R1 tokenizer but several pieces for the
        # base tokenizer, so text after it would not give the same token ids under both models.
        texts = [('our trace (R1-Distill-1.5B)', q, trace.split('</think>')[0], True)]
        for j, r in enumerate(real):
            texts.append((f'real R1 trace {j + 1} (s1K-1.1)', r['question'], r['deepseek_thinking_trajectory'], True))
        texts.append(('control: base model answer', q, base_text, False))

        MAXN = 1024
        R['texts'] = []
        gap_by_token = collections.defaultdict(list)
        for name, question, resp, r1_style in texts:
            umsgs = [{'role': 'user', 'content': question + SUFFIX}]
            dp = dtok.apply_chat_template(umsgs, add_generation_prompt=True, tokenize=False)          # ends '<think>\n'
            bpt = btok.apply_chat_template([{'role': 'user', 'content': question}], add_generation_prompt=True,
                                           tokenize=False) + ('<think>\n' if r1_style else '')
            d_prompt = dtok(dp, add_special_tokens=False)['input_ids']
            b_prompt = btok(bpt, add_special_tokens=False)['input_ids']
            d_resp = dtok(resp, add_special_tokens=False)['input_ids'][:MAXN]
            b_resp = btok(resp, add_special_tokens=False)['input_ids'][:MAXN]
            assert d_resp == b_resp                           # the same response ids under both tokenizers
            nb = per_token_nll(bmodel, b_prompt, b_resp)
            nd = per_token_nll(dmodel, d_prompt, d_resp)
            row = dict(name=name, n=len(d_resp), base=nb.mean().item(), distill=nd.mean().item())
            row['ppl_base'], row['ppl_distill'] = math.exp(row['base']), math.exp(row['distill'])
            log(f'{name:30s} {row["n"]:5d} tokens | base: loss {row["base"]:.3f}, perplexity {row["ppl_base"]:6.2f}'
                f' | distilled: loss {row["distill"]:.3f}, perplexity {row["ppl_distill"]:6.2f}')
            if name.startswith('real R1 trace 1'):
                row['curve_base'], row['curve_distill'] = nb.tolist(), nd.tolist()
                row['tokens'] = [dtok.decode([i]) for i in d_resp]
            if name.startswith('real'):
                for i, a, b in zip(d_resp, nb.tolist(), nd.tolist()):
                    gap_by_token[dtok.decode([i])].append((a, b))
            R['texts'].append(row)
        log('same response token ids under both tokenizers: True (checked for every text)')
        log('tokens of the real R1 traces with the largest gap (seen at least 5 times; mean loss base -> distilled):')
        gaps = [(t, sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v), len(v))
                for t, v in gap_by_token.items() if len(v) >= 5]
        gaps.sort(key=lambda g: g[1] - g[2], reverse=True)
        for t, a, b, n in gaps[:12]:
            log(f'  {t!r:14s} x{n:3d}   {a:5.2f} -> {b:5.2f}')
        R['gaps'] = gaps[:12]
        for w in ['Wait', ' wait', ' Hmm', 'Hmm', 'Okay', ' Let']:
            if w in gap_by_token:
                v = gap_by_token[w]
                log(f'  word {w!r}: x{len(v)}, base {sum(a for a, _ in v) / len(v):.2f}, '
                    f'distilled {sum(b for _, b in v) / len(v):.2f} nats')
    save('part6_loss', R)


if __name__ == '__main__':
    main()
