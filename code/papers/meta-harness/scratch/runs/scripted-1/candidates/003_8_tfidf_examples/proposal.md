# Proposal (scripted proposer, iteration 1)

Parent: 001_few_shot_8 (search accuracy 42.0%, 1139 prompt characters)

Diagnosis from the parent's traces: {"n": 50, "wrong": 29, "invalid": 0, "gold_not_shown": 12, "top_pair": ["card_arrival", "pending_card_payment"], "top_pair_count": 5, "ctx_chars": 1139}

Edit: 12 of 29 wrong answers had no example of the gold label in the prompt; pick the examples by similarity to the message

New config: {'shots': 8, 'select': 'tfidf', 'coverage': False, 'contrastive': False, 'verify': False, 'label_style': 'snake', 'instruction': 'short'}
