"""Chapter 5, second half: build the two small local data sets the experiments use, so every run is repeatable and needs no web search.

1. data/ch5/b/questions.json: 30 multi-step word problems from Northwind Home's buying desk (prices, discounts, VAT, packing, dates,
   currency). Every gold answer is COMPUTED here in Python, not typed by hand, so the grader cannot be wrong about the key.
2. data/ch5/b/suppliers/*.md and data/ch5/b/supplier_facts.json: six fictional kitchenware suppliers, each a short "supplier page"
   with its price, delivery time to the UK, minimum order and risks, plus a few distractor numbers (EU delivery times, prices in
   other currencies) that invite a careless reader to copy the wrong figure. The facts file is the gold key for the hand-off experiment.
All names are invented. Usage: python ch5b_make_data.py"""
import json, math, os, random
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data', 'ch5', 'b')
os.makedirs(os.path.join(OUT, 'suppliers'), exist_ok=True)
rng = random.Random(5)
WD = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def money(x):
    return f'{x:.2f}'


def add_working_days(d, n):
    while n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


Q = []
# five families of multi-step problem, six each, with numbers drawn from a fixed seed. Each needs four to seven steps and has a trap.
for i in range(6):          # A. tiered unit prices, a discount on part of the order only, then VAT
    n = rng.randint(620, 1480)
    t1, t2, t3 = rng.choice([5.20, 4.80, 6.40]), rng.choice([4.10, 3.90, 4.60]), rng.choice([3.25, 3.05, 3.70])
    disc = rng.choice([5, 8, 10])
    goods = min(n, 200) * t1 + min(max(n - 200, 0), 300) * t2 + max(n - 500, 0) * t3
    tier3 = max(n - 500, 0) * t3
    total = (goods - tier3 * disc / 100) * 1.2
    Q.append((f'A supplier prices frying pans in tiers: the first 200 pans cost £{t1:.2f} each, the next 300 cost £{t2:.2f} each, and every pan '
              f'beyond 500 costs £{t3:.2f}. A {disc}% discount applies only to the pans in the top tier (beyond 500). Northwind orders {n} pans, '
              f'and 20% VAT is added to the discounted total. What does Northwind pay in pounds, to the nearest penny?', money(round(total + 1e-9, 2))))
for i in range(6):          # B. stock over weeks: sales, a delivery that arrives part-damaged, reorder point
    stock, sell, deliv, dmg, wk = rng.randint(900, 1400), rng.randint(110, 190), rng.choice([600, 800, 1000]), rng.choice([5, 8, 12.5]), rng.randint(3, 5)
    s, out = stock, None
    for w in range(1, 30):
        s -= sell
        if w == wk:
            s += round(deliv * (1 - dmg / 100))
        if s < 0 and out is None:
            out = w
            break
    Q.append((f'Northwind has {stock} mixing bowls in stock and sells {sell} every week. At the end of week {wk} a delivery of {deliv} bowls arrives, '
              f'but {dmg:g}% of that delivery is damaged and cannot be sold. Sales happen before the delivery in the week it arrives. In which week '
              f'does Northwind first fail to meet the full weekly demand? Answer with the week number.', str(out)))
for i in range(6):          # C. dates: dispatch weekday, sea days, then customs working days with a bank holiday
    d0 = date(2027, rng.choice([3, 4, 7, 11]), rng.randint(1, 28))
    sea, cust = rng.randint(18, 34), rng.randint(3, 7)
    land = d0 + timedelta(days=sea)
    hol = land + timedelta(days=1 + (2 - (land.weekday() + 1)) % 7)   # a Wednesday-ish weekday soon after landing, declared a holiday
    while hol.weekday() >= 5:
        hol += timedelta(days=1)
    d, n = land, cust
    while n:
        d += timedelta(days=1)
        if d.weekday() < 5 and d != hol:
            n -= 1
    Q.append((f'A container leaves the port on {WD[d0.weekday()]} {d0.day} {d0.strftime("%B")} 2027. The sea crossing takes {sea} calendar days. '
              f'Customs clearance then takes {cust} working days, counted from the day after it lands. Working days are Monday to Friday, and '
              f'{WD[hol.weekday()]} {hol.day} {hol.strftime("%B")} is a public holiday. On what date is it cleared? Answer with the day of the month only.',
              str(d.day)))
for i in range(6):          # D. currency, duty on goods only, freight shared by weight between two products
    ua, ub, qa, qb = rng.choice([2.40, 3.15, 5.60]), rng.choice([7.25, 9.90, 6.30]), rng.choice([500, 800, 1200]), rng.choice([300, 400, 600])
    wa, wb, rate, duty, freight = rng.choice([0.4, 0.5, 0.8]), rng.choice([1.5, 2.0, 2.5]), rng.choice([1.25, 1.28, 1.32]), rng.choice([4, 6.5, 8, 12]), rng.choice([480, 600, 900])
    share_b = freight * qb * wb / (qa * wa + qb * wb)
    landed_b = (ub / rate * (1 + duty / 100)) + share_b / qb
    Q.append((f'One shipment holds {qa} whisks quoted at ${ua:.2f} each ({wa:g} kg each) and {qb} casserole dishes quoted at ${ub:.2f} each ({wb:g} kg each). '
              f'Convert at $ {rate:.2f} to £1. Import duty is {duty:g}% of the converted goods value. Freight is £{freight} for the shipment, carries no duty, '
              f'and is shared between the items by weight. What is the landed cost of one casserole dish in pounds, to the nearest penny?',
              money(round(landed_b + 1e-9, 2))))
for i in range(6):          # E. rates: two packers, one starts late, one takes a break
    a, b, early, total, brk = rng.choice([40, 45, 48, 60]), rng.choice([30, 36, 50, 72]), rng.choice([20, 30, 45]), rng.choice([600, 720, 900]), rng.choice([15, 20, 30])
    done, t = 0.0, 0.0
    while done < total - 1e-9:                    # minute by minute: A from 0, B from `early`, A pauses for `brk` minutes starting at minute 120
        ra = 0 if 120 <= t < 120 + brk else a / 60
        rb = b / 60 if t >= early else 0
        step = min(1.0, (total - done) / (ra + rb))
        done += (ra + rb) * step
        t += step
    Q.append((f'Packer A packs {a} parcels an hour and packer B packs {b} an hour. A starts alone at 08:00; B joins {early} minutes later. A takes a '
              f'{brk}-minute break starting at 10:00 while B keeps working. They stop when all {total} parcels are packed. How many minutes after 08:00 do '
              f'they finish? Round to the nearest minute.', str(round(t))))

questions = [{'id': f'q{i + 1:02d}', 'q': q, 'gold': g} for i, (q, g) in enumerate(Q)]
json.dump(questions, open(os.path.join(OUT, 'questions.json'), 'w'), indent=1)

# ---------------------------------------------------------------- suppliers for the hand-off experiment
SUP = [
    dict(id='S1', name='Kestrel Cookware', country='Portugal', price=11.40, cur_note='(€13.30 in the euro price list)', uk_days=9, eu_days=4,
         moq=300, risks=['main kiln is down for repair until February 2027', 'family-owned, single factory']),
    dict(id='S2', name='Lotus Steelworks', country='India', price=7.85, cur_note='(₹830 in the domestic list)', uk_days=34, eu_days=29,
         moq=1200, risks=['ISO 9001 certificate expires in March 2027', 'two late shipments in 2026']),
    dict(id='S3', name='Harbin Enamel Co.', country='China', price=6.10, cur_note='($7.70 FOB Shanghai)', uk_days=41, eu_days=38,
         moq=2000, risks=['coating failed one EU lead-migration test in 2025', 'prices quoted for orders placed before June 2027']),
    dict(id='S4', name='Vardar Kitchen', country='North Macedonia', price=9.95, cur_note='(€11.60 ex works)', uk_days=12, eu_days=6,
         moq=500, risks=['owner is in a contract dispute with a former distributor', 'no UKCA marking yet']),
    dict(id='S5', name='Brightline Housewares', country='Vietnam', price=8.30, cur_note='($10.45 FOB Hai Phong)', uk_days=38, eu_days=33,
         moq=1000, risks=['factory sits in a typhoon flood zone', 'new owner since 2026']),
    dict(id='S6', name='Oakmere Forge', country='United Kingdom', price=14.75, cur_note='(trade price before VAT)', uk_days=3, eu_days=8,
         moq=150, risks=['capacity limited to 4,000 units a month', 'steel surcharge may add up to 6%']),
]
FILL = ['The company was founded in {y} and exports to {k} countries.', 'Its catalogue lists {c} kitchenware lines, mostly pans and utensils.',
        'Samples are sent free for orders above the minimum.', 'Payment terms are 30% on order and 70% before shipping.']
for s in SUP:
    y, k, c = rng.randint(1958, 2016), rng.randint(6, 40), rng.randint(40, 260)
    page = [f'# {s["name"]} ({s["country"]})', '',
            ' '.join(f.format(y=y, k=k, c=c) for f in FILL[:2]),
            f'Price for the reference 24 cm frying pan: £{s["price"]:.2f} per unit, delivered duty paid to the UK {s["cur_note"]}.',
            f'Delivery: {s["eu_days"]} days to an EU warehouse; {s["uk_days"]} days door to door to a UK warehouse.',
            f'Minimum order quantity: {s["moq"]:,} units per order.',
            'Known risks: ' + '; '.join(s['risks']) + '.',
            ' '.join(FILL[2:])]
    open(os.path.join(OUT, 'suppliers', f'{s["id"]}.md'), 'w').write('\n'.join(page) + '\n')
facts = [{'id': s['id'], 'name': s['name'], 'price_gbp': s['price'], 'uk_days': s['uk_days'], 'moq': s['moq'], 'risks': s['risks']} for s in SUP]
json.dump(facts, open(os.path.join(OUT, 'supplier_facts.json'), 'w'), indent=1)
print(f'{len(questions)} questions -> data/ch5/b/questions.json; {len(SUP)} supplier pages -> data/ch5/b/suppliers/; gold -> supplier_facts.json')
for q in questions[:30:6]:
    print(q['id'], q['gold'], '|', q['q'][:110])
