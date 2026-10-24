"""Chapter 6, second half: the small fake Northwind Home data set used by ch6b_judge.py and ch6b_trajectory.py.
Writes data/ch6/b/policy.md, orders.json and replies.json. No model call: every reply is written from a template, and its
label (pass or fail, and which defect) is set by construction, so the 'human labels' are known exactly.

replies.json holds 80 support replies in two sets of 40:
  set 'calib' (20 pass, 20 fail): the labelled set we use to measure the judge (TPR, TNR);
  set 'prod'  (28 pass, 12 fail): stands in for a week of production replies whose true pass rate (70%) the judge must estimate.
Each failing reply has exactly one planted defect from DEFECTS. Deterministic (seed 6)."""
import json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data', 'ch6', 'b')
os.makedirs(OUT, exist_ok=True)
TODAY = '2026-10-20'

POLICY = """# Northwind Home support policy (excerpt)
1. Returns: an item can be returned within 30 days of delivery for a full refund to the original payment method.
   Refunds reach the customer in 5 to 7 business days after the return arrives.
2. Return shipping: free for damaged or wrong items. For other returns, a 6.95 USD label fee is deducted from the refund.
3. Damaged on arrival: report within 14 days of delivery with a photo. We send a replacement or a full refund; no return needed.
4. Cancellation: an order can be cancelled only before it ships. After it ships, the customer must use a return.
5. Delivery dates: never promise a delivery date. Give the carrier's tracking status only.
6. Tone: polite and direct. Answer the customer's question in the first two sentences."""

ITEMS = [('Oak side table', 129.00), ('Linen duvet cover', 89.50), ('Ceramic table lamp', 64.00), ('Wool throw blanket', 72.00),
         ('Bamboo bath mat', 24.99), ('Walnut cutting board', 45.00), ('Velvet cushion set', 58.00), ('Glass storage jars', 32.50),
         ('Rattan laundry basket', 49.00), ('Cast-iron skillet', 39.95)]
FIRST = ['Maya', 'Omar', 'Lena', 'Jonas', 'Priya', 'Tom', 'Ana', 'Ken', 'Sara', 'Luis']

DEFECTS = {
    'wrong_amount': 'states a refund amount that does not follow the policy',
    'policy_violation': 'grants something the policy forbids (refund outside 30 days, cancel after shipping)',
    'invented_promise': 'promises something not in the policy (a delivery date, free shipping, a discount)',
    'no_answer': 'does not answer the question; generic text',
    'wrong_fact': 'gives a wrong order status or wrong item',
    'rude': 'dismissive or blaming tone',
    'wrong_timeline': 'gives a refund time or rule detail that differs from the policy (2 to 3 days, 14-day window)',
    'subtle_amount': 'an amount that is off by a policy detail (wrong fee, fee on a free return)',
}


def days_between(a, b):
    from datetime import date
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def make_orders(rng, n):
    orders = []
    for i in range(n):
        item, price = ITEMS[i % len(ITEMS)]
        kind = ['status', 'return_ok', 'return_late', 'damaged', 'cancel_ok', 'cancel_shipped'][i % 6]
        o = {'order_id': f'NW-{4100 + i}', 'customer': FIRST[i % len(FIRST)], 'item': item, 'price': price, 'kind': kind}
        if kind == 'status':
            o.update(status='shipped', tracking='in transit, last scan at the regional hub', delivered=None)
        elif kind == 'damaged':
            o.update(status='delivered', delivered=['2026-10-12', '2026-10-15', '2026-10-17'][i % 3])
        elif kind == 'return_ok':
            o.update(status='delivered', delivered=['2026-09-21', '2026-10-09', '2026-09-24'][i % 3])      # 29, 11 and 26 days ago
        elif kind == 'return_late':
            o.update(status='delivered', delivered=['2026-09-19', '2026-08-30', '2026-09-18'][i % 3])      # 31, 51 and 32 days ago
        elif kind == 'cancel_ok':
            o.update(status='processing', delivered=None)
        else:
            o.update(status='shipped', tracking='in transit', delivered=None)
        orders.append(o)
    return orders


MESSAGES = {
    'status': 'Hi, where is my order {id}? It has been a while.',
    'return_ok': 'I want to return the {item} from order {id}. It does not fit my room. How much will I get back?',
    'return_late': 'Can I still return the {item} from order {id}? I changed my mind.',
    'damaged': 'The {item} in order {id} arrived cracked. I have a photo. What can you do?',
    'cancel_ok': 'Please cancel order {id}, I ordered the wrong {item}.',
    'cancel_shipped': 'I need to cancel order {id} for the {item} right away.',
}


def good_reply(o):
    n, it, oid, p = o['customer'], o['item'], o['order_id'], o['price']
    k = o['kind']
    if k == 'status':
        return f"Hi {n}, order {oid} ({it}) has shipped. The carrier shows it {o['tracking']}. I can't give an exact delivery date, but the tracking link in your shipping email updates as it moves."
    if k == 'return_ok':
        return f"Hi {n}, yes, the {it} from order {oid} is within our 30-day return window. Your refund will be {p - 6.95:.2f} USD: the {p:.2f} USD price minus the 6.95 USD return label fee. It reaches your card 5 to 7 business days after the return arrives."
    if k == 'return_late':
        return f"Hi {n}, I'm sorry, but order {oid} was delivered on {o['delivered']}, which is more than 30 days ago, so the {it} can no longer be returned. If something is wrong with the item itself, reply with a photo and we will take a look."
    if k == 'damaged':
        return f"Hi {n}, I'm sorry the {it} arrived cracked. Please reply with the photo, and we will send a replacement or a full refund of {p:.2f} USD, whichever you prefer. You don't need to send the item back."
    if k == 'cancel_ok':
        return f"Hi {n}, done: order {oid} for the {it} had not shipped yet, so I have cancelled it. The full {p:.2f} USD goes back to your original payment method."
    return f"Hi {n}, order {oid} has already shipped, so I can't cancel it. When it arrives you can return the {it} within 30 days; the refund is the price minus a 6.95 USD return label fee."


def bad_reply(o, defect, rng):
    n, it, oid, p, k = o['customer'], o['item'], o['order_id'], o['price'], o['kind']
    g = good_reply(o)
    if defect == 'wrong_amount':
        if k == 'return_ok':
            return g.replace(f'{p - 6.95:.2f} USD: the {p:.2f} USD price minus the 6.95 USD return label fee', f'{p:.2f} USD, the full price')
        if k in ('damaged', 'cancel_ok'):
            return g.replace(f'{p:.2f} USD', f'{p - 6.95:.2f} USD (the price minus our 6.95 USD handling fee)', 1)
        return g + f' You will get {p:.2f} USD back in full.'
    if defect == 'policy_violation':
        if k == 'return_late':
            return f"Hi {n}, no problem. I've opened a return for the {it} from order {oid}; you will get {p - 6.95:.2f} USD back once it arrives."
        if k == 'cancel_shipped':
            return f"Hi {n}, I've cancelled order {oid} for the {it}. The full {p:.2f} USD will be back on your card in 5 to 7 business days."
        if k == 'damaged':
            return g.replace("You don't need to send the item back.", 'Please first ship the item back to us at your own cost; we refund after it arrives.')
        return g + ' I have also added a 15% discount to your next order for the trouble.'
    if defect == 'invented_promise':
        if k in ('status', 'cancel_shipped'):
            return g + ' It will be with you by Thursday at the latest.'
        return g + ' Return shipping is free for you this time.' if k == 'return_ok' else g + ' I have also added a 20 USD credit to your account.'
    if defect == 'wrong_timeline':
        if k == 'return_ok':
            return g.replace('5 to 7 business days', '2 to 3 business days')
        if k == 'return_late':
            return g.replace('more than 30 days ago', 'more than 30 days ago (our window is 21 days)')
        if k in ('cancel_shipped',):
            return g.replace('within 30 days', 'within 14 days')
        if k == 'damaged':
            return g.replace('Please reply with the photo', 'Please reply with the photo within 30 days')
        return g + ' The money is usually back on your card the same day.'
    if defect == 'subtle_amount':
        if k == 'return_ok':
            return g.replace(f'{p - 6.95:.2f} USD: the {p:.2f} USD price minus the 6.95 USD', f'{p - 9.95:.2f} USD: the {p:.2f} USD price minus the 9.95 USD')
        if k == 'damaged':
            return g.replace(f'full refund of {p:.2f} USD', f'refund of {p - 6.95:.2f} USD')
        if k == 'cancel_shipped':
            return g.replace('minus a 6.95 USD return label fee', 'with free return shipping')
        if k == 'cancel_ok':
            return g.replace(f'The full {p:.2f} USD', f'{p - 6.95:.2f} USD (after the 6.95 USD fee)')
        return g + f' The refund would be {p:.2f} USD.'
    if defect == 'no_answer':
        return f"Hi {n}, thank you for contacting Northwind Home! We value every customer. Our team handles requests like yours with care, and you can find more information in our help centre. Have a lovely day!"
    if defect == 'wrong_fact':
        if k == 'status':
            return f"Hi {n}, order {oid} ({it}) is still being processed in our warehouse and has not shipped yet. You will get an email when it ships."
        if k == 'cancel_ok':
            return f"Hi {n}, order {oid} has already shipped, so I can't cancel it. When it arrives you can return it within 30 days."
        other = ITEMS[(ITEMS.index((it, p)) + 3) % len(ITEMS)][0]
        return g.replace(it, other)
    if defect == 'rude':
        return {'status': f"Order {oid} has shipped. Check the tracking link we already emailed you; it's all there.",
                'return_ok': f"You should measure your room before ordering. Anyway, the {it} is within 30 days, so you get {p - 6.95:.2f} USD back after the 6.95 USD label fee.",
                'return_late': f"Order {oid} was delivered on {o['delivered']}. That's over 30 days. You should have decided sooner. No return.",
                'damaged': f"Items rarely break in our packaging, so are you sure it wasn't you? Fine, send the photo and we'll send a replacement or refund {p:.2f} USD.",
                'cancel_ok': f"Please read your order before you place it next time. Order {oid} is cancelled; {p:.2f} USD goes back to your card.",
                'cancel_shipped': f"It has shipped, obviously we can't cancel it now. Return it when it comes; we deduct 6.95 USD."}[k]
    raise ValueError(defect)


# which defects make sense for which kind of request
FITS = {'status': ['invented_promise', 'wrong_fact', 'wrong_timeline', 'no_answer'],
        'return_ok': ['subtle_amount', 'wrong_timeline', 'invented_promise', 'wrong_amount', 'rude'],
        'return_late': ['policy_violation', 'wrong_timeline', 'wrong_fact', 'policy_violation'],
        'damaged': ['subtle_amount', 'policy_violation', 'wrong_timeline', 'invented_promise'],
        'cancel_ok': ['subtle_amount', 'wrong_fact', 'invented_promise', 'wrong_timeline'],
        'cancel_shipped': ['policy_violation', 'subtle_amount', 'wrong_timeline', 'invented_promise']}


def terse(o):
    """A correct reply in a short, informal style: a vague judge may mark it down although nothing in it is wrong."""
    n, it, oid, p, k = o['customer'], o['item'], o['order_id'], o['price'], o['kind']
    return {'status': f"Hi {n}. {oid} shipped and is {o.get('tracking', 'in transit')}. No delivery date from the carrier yet, sorry.",
            'return_ok': f"Hi {n}. Yes, you're inside 30 days. Refund: {p - 6.95:.2f} USD (6.95 USD label fee). 5 to 7 business days after it reaches us.",
            'return_late': f"Hi {n}. Sorry, no: delivered {o['delivered']}, over our 30-day limit.",
            'damaged': f"Hi {n}. Sorry about that. Send the photo; we'll refund {p:.2f} USD in full or replace it, your choice. Keep the item.",
            'cancel_ok': f"Hi {n}. Cancelled. {p:.2f} USD back to your card.",
            'cancel_shipped': f"Hi {n}. Too late to cancel, {oid} has shipped. Return it within 30 days of delivery instead (6.95 USD label fee)."}[k]


def main():
    rng = random.Random(6)
    orders = make_orders(rng, 80)
    replies = []
    for setname, start, n_pass in (('calib', 0, 20), ('prod', 40, 28)):
        idx = list(range(start, start + 40))
        rng.shuffle(idx)
        passing = set(idx[:n_pass])
        for i in range(start, start + 40):
            o = orders[i]
            msg = MESSAGES[o['kind']].format(id=o['order_id'], item=o['item'].lower())
            if i in passing:
                text, label, defect = (terse(o) if i % 2 else good_reply(o)), 'pass', None
            else:
                defect = FITS[o['kind']][(i * 7) % len(FITS[o['kind']])]
                text, label = bad_reply(o, defect, rng), 'fail'
            replies.append({'id': f'R{i + 1:02d}', 'set': setname, 'order_id': o['order_id'], 'message': msg, 'reply': text,
                            'label': label, 'defect': defect})
    json.dump([{k: v for k, v in o.items() if k != 'kind'} for o in orders], open(os.path.join(OUT, 'orders.json'), 'w'), indent=1)
    open(os.path.join(OUT, 'policy.md'), 'w').write(POLICY + '\n')
    json.dump(replies, open(os.path.join(OUT, 'replies.json'), 'w'), indent=1)
    from collections import Counter
    for s in ('calib', 'prod'):
        rs = [r for r in replies if r['set'] == s]
        print(s, len(rs), 'replies;', sum(r['label'] == 'pass' for r in rs), 'pass;', dict(Counter(r['defect'] for r in rs if r['defect'])))


if __name__ == '__main__':
    main()
