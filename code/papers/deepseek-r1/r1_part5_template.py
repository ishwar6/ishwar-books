"""Part 5, §2.3.1: the readable output format |special_token|<reasoning_process>|special_token|<summary>,
as it shows in the chat template of the released DeepSeek-R1-Distill-Qwen-1.5B.

Method 1: our own Python function that builds the prompt string for user and assistant turns.
Method 2: the tokenizer's own Jinja template (tokenizer.apply_chat_template).
Then: a check that both give the same string, and the token ids of <think> and </think>.
Only the tokenizer is loaded (no model weights), so this needs no heavy-job lock."""
import common
from transformers import AutoTokenizer

MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
log = common.Log('r1_part5_template')
tok = AutoTokenizer.from_pretrained(MODEL)

BOS, EOS = '<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>'
USER, ASSISTANT = '<｜User｜>', '<｜Assistant｜>'


# ---------------------------------------------------------------- Method 1: the template in plain Python
def r1_prompt(messages, add_generation_prompt=False):
    """Build the prompt text the way the released template does (system, user and assistant turns; no tools)."""
    system = ''.join(m['content'] for m in messages if m['role'] == 'system')
    out = BOS + system                                       # the system text goes first, with no tag
    for m in messages:
        if m['role'] == 'user':
            out += USER + m['content']
        elif m['role'] == 'assistant':
            content = m['content'].split('</think>')[-1]    # an old answer keeps only its summary
            out += ASSISTANT + content + EOS
    if add_generation_prompt:
        out += ASSISTANT + '<think>\n'                       # the model's turn opens its thinking
    return out


chat = [
    {'role': 'user', 'content': 'What is 15% of 80?'},
    {'role': 'assistant', 'content': '<think>\n10% of 80 is 8, and 5% is 4, so 12.\n</think>\n\nThe answer is **12**.'},
    {'role': 'user', 'content': 'And 15% of 60?'},
]

log('two pieces of tokenizer.chat_template (its full length is', len(tok.chat_template), 'characters):')
log('  ...' + tok.chat_template[tok.chat_template.find("{% set content = message['content'] %}"):][:150] + '...')
log('  ...' + tok.chat_template[-95:])
log('')
ours = r1_prompt(chat, add_generation_prompt=True)
lib = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
log('Method 1 (our function):')
log(repr(ours))
log('Method 2 (tokenizer.apply_chat_template):')
log(repr(lib))
log('same string:', ours == lib)
log('')
log('the thinking of the first answer is in the prompt:', '10% of 80 is 8' in lib)
log('')
for t in ['<think>', '</think>', USER, ASSISTANT, EOS]:
    ids = tok.encode(t, add_special_tokens=False)
    log(f'  {t!r:28s} -> token ids {ids}')
log('')
n = len(tok(lib, add_special_tokens=False).input_ids)
log(f'the prompt is {n} tokens; generation starts right after "<think>\\n"')
common.save('r1_part5_template', dict(same=ours == lib, prompt=lib, n_tokens=n,
                                      think_ids=tok.encode('<think>', add_special_tokens=False),
                                      end_think_ids=tok.encode('</think>', add_special_tokens=False)))
