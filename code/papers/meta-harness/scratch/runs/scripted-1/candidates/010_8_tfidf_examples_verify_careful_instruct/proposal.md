# Proposal (scripted proposer, iteration 4)

Parent: 006_8_tfidf_examples_verify (search accuracy 56.0%, 2318 prompt characters)

Diagnosis from the parent's traces: {"n": 50, "wrong": 22, "invalid": 0, "gold_not_shown": 4, "top_pair": ["card_arrival", "pending_card_payment"], "top_pair_count": 3, "ctx_chars": 2318}

Edit: tell the model that several labels are close

New config: {'shots': 8, 'select': 'tfidf', 'coverage': False, 'contrastive': False, 'verify': True, 'label_style': 'snake', 'instruction': 'careful'}
