# Proposal (scripted proposer, iteration 2)

Parent: 003_8_tfidf_examples (search accuracy 54.0%, 1177 prompt characters)

Diagnosis from the parent's traces: {"n": 50, "wrong": 23, "invalid": 0, "gold_not_shown": 4, "top_pair": ["card_arrival", "pending_card_payment"], "top_pair_count": 4, "ctx_chars": 1177}

Edit: the most common mistake is card_arrival -> pending_card_payment (4 times); show close examples with different labels side by side

New config: {'shots': 8, 'select': 'tfidf', 'coverage': False, 'contrastive': True, 'verify': False, 'label_style': 'snake', 'instruction': 'short'}
