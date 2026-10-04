# BERT, explained (code)

Code and results for the six-part breakdown of *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
(Devlin, Chang, Lee, Toutanova; arXiv:1810.04805) at https://ishwarj.com/papers/bert/

Every number in the articles that does not come from the paper comes from one of these scripts, and the exact output is saved in `results/`.

| File | What it does |
|---|---|
| `bert_part1.py` | Part 1: GPT-2 (left side only) vs BERT (both sides) on the same fill-in-the-blank sentences |
| `bert_part2.py` | Part 2: parameter counts by formula and by counting (BERT-base, BERT-large on the meta device), the 30,522-entry vocabulary, WordPiece vs the real tokenizer, the input embedding sum, "bank" in four sentences, the pooler, layer 1 recomputed by hand |
| `bert_part3_seeitself.py` | Part 3: three tiny Transformers on WikiText-2 showing that a two-layer "both sides" language model cheats (it sees itself) |
| `bert_part3_mlm.py` | Part 3: the 15% / 80-10-10 masking procedure measured on WikiText-103, "my dog is hairy" on the real model, masked-LM accuracy, MLM + NSP loss on one batch |
| `bert_part3_nsp.py` | Part 3: the pre-trained NSP head on the paper's examples and on 1,000 real pairs, raw [CLS] cosine similarities (footnote 6) |
| `bert_part3_schedule.py` | Part 3: batch and epoch arithmetic, the learning-rate schedule, GELU vs ReLU, attention cost at 128 vs 512 tokens |
| `bert_part3_data.py` | Part 3: WikiText loading and clean-up shared by the Part 3 scripts |
| `bert_part4_heads.py` | Part 4: the GLUE loss -log softmax(C W^T) and the SWAG head checked against the library |
| `bert_part4_finetune.py` | Part 4: a real fine-tune of bert-base-uncased on SST-2 and MRPC (Apple GPU) |
| `bert_part4_squad.py` | Part 4: our own span search (S.T_i + E.T_j) and null score on public SQuAD v1.1 / v2.0 checkpoints, scored on the full dev sets |
| `bert_part5.py` | Part 5: Table 5 differences, Table 6 parameter counts, masked-LM perplexity on WikiText-2 |
| `bert_part5_ner.py` | Part 5: frozen bert-base-cased features for CoNLL-2003 NER, six feature choices, a BiLSTM tagger each |
| `bert_part6.py` | Part 6: the 512-token limit, raw [CLS] vs a fine-tuned sentence model, two masks at once, NER with a public checkpoint |
| `common.py` | a log that prints and saves to `results/<name>_stdout.txt`, and JSON results |
| `paper_shots.py`, `shots_part*.py` | download the paper, highlight phrases and crop excerpts into `public/img/papers/bert/` |
| `figlib.py`, `bertfig.py`, `figs_part*.py` | draw the SVG figures from the results (theme-aware CSS classes, no hard-coded colours) |
| `assemble.py` | turns `partN.md` into `content/papers/bert/<slug>.md`, inserting the figures; checks for missing images |
| `check_md.mjs` | renders a page's maths with KaTeX and reports errors (run from the repo root) |
| `results/` | the JSON and text output every number in the articles comes from |

```bash
pip install torch transformers datasets pymupdf
python bert_part1.py            # and the other bert_part*.py scripts
python shots_part1.py           # paper excerpts (needs pymupdf)
python figs_part1.py && python assemble.py 1
```
