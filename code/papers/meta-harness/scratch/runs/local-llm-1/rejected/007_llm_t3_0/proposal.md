# Proposal (local LLM proposer Qwen/Qwen2.5-1.5B-Instruct, iteration 3, sample 0)

IDEA: Improve the diversity of prompts to better capture the nuances of customer messages.

Files it asked for: "Idea:\n- **Change the prompt to focus more on the customer's experience and sentiment rather than just banking details**.\n- This will make the model pay closer attention to how the customer feels about their interactions with the bank.\n\n```python\nimport hlib\nfrom typing import List\n\nclass NewHarness("
Files read: ['004_llm_t1_1/harness.py', '004_llm_t1_1/traces.jsonl'] (fallback: it named no real file); also shown: 004_llm_t1_1/harness.py
Self-check attempts: ['passed']

## Full reply

IDEA: Improve the diversity of prompts to better capture the nuances of customer messages.

```python
import random
from hlib import TfIdf, parse_label, show_label

CONFIG = {'shots': 5, 'select': 'random', 'coverage': True, 'contrastive': False, 'verify': False, 'label_style': 'words', 'instruction': 'long'}

INSTRUCTIONS = {
    'long': 'Classify the bank customer message into exactly one of the labels by considering the entire message content, context and implications.',
}

class Harness:
    def __init__(self, llm, labels):
        self.llm, self.labels = llm, labels
        self.memory = []         # every (text, label) pair seen so far, in order
        self.index = None        # TF-IDF index over the memory, built when first needed

    def update(self, text, label):
        """See one labeled example (the online stream). Store it; rebuild the index later."""
        self.memory.append((text, label))
        self.index = None

    # ---- helpers ------------------------------------------------------------------------------
    def name(self, label):
        return show_label(label) if CONFIG['label_style'] == 'words' else label

    def nearest(self, text, k, keep=lambda label: True):
        """The k stored examples most similar to text (TF-IDF cosine) whose label passes keep()."""
        if self.index is None:
            self.index = TfIdf([t for t, _ in self.memory])
        order = self.index.top(text, len(self.memory))
        return [self.memory[i] for i in order if keep(self.memory[i][1])][:k]

    def examples(self, text):
        k, how = CONFIG['shots'], CONFIG['select']
        if k == 0 or not self.memory:
            return []
        if how == 'first':
            return self.memory[:k]
        if how == 'random':
            return random.Random(0).sample(self.memory, min(k, len(self.memory)))
        return self.nearest(text, k)[::-1]   # tfidf: most similar example last, next to the question

    def coverage_block(self, text):
        return [self.nearest(text, 1, keep=lambda l: True)[0]
                for label in self.labels if any(l == label for _, l in self.memory)]

    def contrastive_block(self, text):
        out, seen = [], set()
        for t, l in self.nearest(text, 8):   # the 8 nearest; keep the first example of each label
            if l not in seen:
                seen.add(l)
                out.append((t, l))
        return out[:4]

    def prompt(self, text, shown, extra=''):
        lines = [INSTRUCTIONS[CONFIG['instruction']], '', 'Labels:', ]
        lines += ['- ' + self.name(l) for l in self.labels]
        if shown:
            lines += ['', 'Examples:']
            lines += [f'Message: {t}\nLabel: {self.name(l)}' for t, l in shown]
        if extra:
            lines += ['', extra]
        lines += ['', f'Message: {text}', 'Answer with the label only.', 'Label:']
        return '\n'.join(lines)

    # ---- the two methods the evaluator calls -----------------------------------------------------
    def predict(self, text):
        shown = self.examples(text)
        if CONFIG['coverage'] and self.memory:
            cover = self.coverage_block(text)
            shown = cover + [e for e in shown if e not in cover]
        if CONFIG['contrastive'] and self.memory:
            shown = shown + [e for e in self.contrastive_block(text) if e not in shown]
        draft = parse_label(self.llm(self.prompt(text, shown)), self.labels) or self.labels[0]
        if not CONFIG['verify'] or len(self.memory) < 5:
            return draft
        confirm = self.nearest(text, 3, keep=lambda l: l == draft)
        challenge = self.nearest(text, 3, keep=lambda l: l != draft)
        note = (f'A first answer was "{self.name(draft)}". Check it. Examples with that label and similar'
                ' examples with other labels are shown above. Keep it or change it.')
        final = self.llm(self.prompt(text, confirm + challenge, extra=note))
        return parse_label(final, self.labels) or draft

if __name__ == "__main__":
    # Example usage
    llm = lambda prompt: 'answer here'  # Replace with your actual LLM function
    labels = ['activate_my_card', 'card_arrival', 'card_delivery_estimate', 'card_not_working',
              'card_payment_fee_charged', 'card_payment_not_recognised', 'compromised_card',
              'declined_card_payment', 'lost_or_stolen_card', 'pending_card_payment']

    harness = Harness(llm, labels)
    print(harness.predict("I need help because I've noticed something strange about my recent transactions."))

```
