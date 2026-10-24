"""Chapter 5 (first half): build the small, fake document set for Northwind's supplier question, and its known answer.
Everything is fictional: 24 kitchenware suppliers, each with a long catalogue page, a news feed, a shipping table and Northwind's
buying rules. The known answer (gold.json) is computed here, in code, from the same facts the agents must find with their tools.
Usage: python ch5a_data.py   ->  data/ch5/a/{requirements.md, suppliers/S01.md ..., news.json, shipping.json, gold.json}"""
import json, os, random
from common import Log

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data', 'ch5', 'a')
log = Log('ch5a_data')

# ---------------------------------------------------------------- Northwind's buying rules (the lead agent reads these with a tool)
FX = {'GBP': 1.0, 'EUR': 0.85, 'USD': 0.78, 'CNY': 0.11, 'INR': 0.0092, 'TRY': 0.022, 'PLN': 0.20}
EU = {'Portugal', 'Poland', 'Spain', 'Germany', 'Netherlands', 'Italy', 'Czechia'}
DUTY = 0.04            # import duty on FOB goods from outside the EU and the UK, as a share of the basket price in GBP
MAX_DAYS = 60          # production lead time + sea or road transit to Felixstowe
MAX_MOQ = 1000         # minimum order quantity, units

REQUIREMENTS = f"""# Northwind Home: rules for choosing new kitchenware suppliers for 2027

Approved by the buying team. Apply every rule exactly as written.

1. Reference basket. Compare suppliers on the price of one reference basket: a 28 cm frying pan, a 20 cm chef's knife and a set of
   three mixing bowls. Every supplier page quotes a "reference basket price" in its own currency.
2. Currency. Convert to GBP with finance's fixed planning rates: 1 EUR = 0.85 GBP, 1 USD = 0.78 GBP, 1 CNY = 0.11 GBP,
   1 INR = 0.0092 GBP, 1 TRY = 0.022 GBP, 1 PLN = 0.20 GBP. Do not use market rates.
3. Landed cost per basket (GBP). If the price is FOB (free on board at the supplier's port), add the freight per basket from the
   shipping table for that port, and add import duty of 4% of the converted basket price when the goods come from outside the EU and
   the UK. If the price is DDP (delivered duty paid to our warehouse), the converted price already is the landed cost.
4. Delivery time. Delivery days = the supplier's production lead time + the transit days from its port in the shipping table.
   Reject any supplier with delivery days above {MAX_DAYS}.
5. Minimum order. Reject any supplier whose minimum order quantity is above {MAX_MOQ:,} units.
6. Safety and ethics. Reject any supplier with an OPEN product safety recall or a suspended ethical audit in the news feed.
   A recall that the news says is closed or resolved is not a reason to reject, but report it as a risk.
   Check names carefully: different companies can have similar names.
7. Choice. Among the suppliers that pass rules 4 to 6, choose the three with the lowest landed cost per basket.
   For each, report the landed cost (GBP, two decimals), the delivery days and the main risks.
"""

# ---------------------------------------------------------------- the 24 suppliers: the facts that decide the answer
# id, name, city, country, port, currency, basket price, incoterm, lead days, MOQ, factories
SUP = [
    ('S01', 'Duro Cookware', 'Porto', 'Portugal', 'Leixoes', 'EUR', 49.0, 'FOB', 14, 300, 2),
    ('S02', 'Kestrel Steelworks', 'Sheffield', 'United Kingdom', 'UK road', 'GBP', 47.5, 'DDP', 10, 200, 1),
    ('S03', 'Sun Harbour Kitchen', 'Ningbo', 'China', 'Ningbo', 'CNY', 270.0, 'FOB', 20, 500, 3),        # cheapest; OPEN recall
    ('S04', 'Vistula Enamel', 'Gdansk', 'Poland', 'Gdansk', 'PLN', 205.0, 'FOB', 18, 400, 1),
    ('S05', 'Anatolia Metal', 'Izmir', 'Turkey', 'Izmir', 'TRY', 1650.0, 'FOB', 15, 600, 2),
    ('S06', 'Pearl River Housewares', 'Foshan', 'China', 'Yantian', 'CNY', 330.0, 'FOB', 25, 800, 4),
    ('S07', 'Hanil Kitchen', 'Busan', 'South Korea', 'Busan', 'USD', 40.0, 'FOB', 35, 500, 1),            # cheap; 35 + 40 days > 60
    ('S08', 'Lumen Homeware', 'Porto', 'Portugal', 'Leixoes', 'EUR', 47.0, 'FOB', 16, 300, 1),          # name trap: recall is Lumen Housewares (US)
    ('S09', 'Ganges Steel Craft', 'Moradabad', 'India', 'Mundra', 'INR', 3900.0, 'FOB', 21, 500, 2),      # resolved recall: eligible, a risk
    ('S10', 'Bremer Kuche', 'Bremen', 'Germany', 'Hamburg', 'EUR', 55.0, 'FOB', 12, 250, 1),
    ('S11', 'Golden Wok Industrial', 'Jieyang', 'China', 'Yantian', 'CNY', 280.0, 'FOB', 18, 2000, 5),   # cheap; MOQ 2,000
    ('S12', 'Atlas Kitchen Trading', 'Valencia', 'Spain', 'UK warehouse', 'GBP', 39.9, 'DDP', 24, 400, 2), # DDP: price is landed
    ('S13', 'Saigon Hearth', 'Binh Duong', 'Vietnam', 'Ho Chi Minh', 'USD', 47.0, 'FOB', 18, 500, 1),
    ('S14', 'Moravia Ceramics', 'Brno', 'Czechia', 'Hamburg', 'EUR', 52.0, 'FOB', 20, 300, 1),
    ('S15', 'Delta Cookware', 'Rotterdam', 'Netherlands', 'Rotterdam', 'EUR', 58.0, 'FOB', 8, 150, 1),
    ('S16', 'Lotus Pan Works', 'Ningbo', 'China', 'Ningbo', 'CNY', 365.0, 'FOB', 22, 600, 2),
    ('S17', 'Iberia Forge', 'Bilbao', 'Spain', 'Bilbao', 'EUR', 51.0, 'FOB', 14, 300, 1),
    ('S18', 'Mumbai Kitchen Exports', 'Mumbai', 'India', 'Nhava Sheva', 'INR', 4500.0, 'FOB', 24, 700, 2),
    ('S19', 'Bosphorus Housewares', 'Istanbul', 'Turkey', 'Istanbul', 'TRY', 1850.0, 'FOB', 12, 500, 1),  # suspended ethical audit
    ('S20', 'Tuscan Copper', 'Florence', 'Italy', 'Livorno', 'EUR', 62.0, 'FOB', 15, 200, 1),
    ('S21', 'Yangtze Steel Home', 'Wuhan', 'China', 'Shanghai', 'CNY', 340.0, 'FOB', 26, 900, 3),
    ('S22', 'Carpathia Kitchen', 'Krakow', 'Poland', 'Gdansk', 'PLN', 240.0, 'FOB', 14, 350, 1),
    ('S23', 'Andes Housewares', 'Monterrey', 'Mexico', 'Altamira', 'USD', 52.0, 'FOB', 20, 400, 2),
    ('S24', 'Pacific Rim Kitchen', 'Kaohsiung', 'Taiwan', 'Kaohsiung', 'USD', 46.0, 'FOB', 30, 500, 2),
]

SHIPPING = {   # port -> (transit days to Felixstowe or to the warehouse, freight per basket in GBP)
    'Leixoes': (5, 1.10), 'UK road': (2, 0.00), 'Ningbo': (34, 2.10), 'Gdansk': (4, 0.90), 'Izmir': (11, 1.50),
    'Yantian': (36, 2.20), 'Busan': (40, 2.30), 'Mundra': (28, 2.40), 'Hamburg': (3, 0.80), 'UK warehouse': (1, 0.00),
    'Ho Chi Minh': (38, 2.40), 'Rotterdam': (2, 0.70), 'Bilbao': (4, 1.00), 'Nhava Sheva': (29, 2.50), 'Istanbul': (12, 1.60),
    'Livorno': (9, 1.30), 'Shanghai': (35, 2.10), 'Altamira': (24, 2.60), 'Kaohsiung': (37, 2.30),
}

# ---------------------------------------------------------------- the news feed: what decides rule 6, plus noise
NEWS = [
    ('N01', '2026-08-14', 'Product safety recall: Sun Harbour Kitchen non-stick pans',
     'UK trading standards announced a recall of 28 cm non-stick frying pans made by Sun Harbour Kitchen of Ningbo, after tests found '
     'coating flaking at normal cooking temperatures. The recall is open; the company says a fix is being tested and no end date is set.'),
    ('N02', '2026-09-02', 'Lumen Housewares recalls electric kettles in the United States',
     'Lumen Housewares, a small-appliance brand based in Columbus, Ohio, recalled 40,000 electric kettles after reports of faulty '
     'thermostats. The recall is open. Lumen Housewares is not connected to any European cookware company of a similar name.'),
    ('N03', '2025-11-20', 'Ganges Steel Craft closes knife-handle recall',
     'Ganges Steel Craft of Moradabad said its recall of a batch of chef\'s knives with loose handles, opened in March 2025, is now '
     'closed. Regulators confirmed the corrective action. The company changed its rivet supplier.'),
    ('N04', '2026-07-30', 'Ethical audit suspended at Bosphorus Housewares',
     'An independent auditor suspended the ethical trade certificate of Bosphorus Housewares, Istanbul, after finding excessive '
     'overtime at its stamping plant. The suspension stays in place until a re-audit, not expected before 2027.'),
    ('N05', '2026-06-11', 'Port congestion eases at Yantian',
     'Container waiting times at Yantian fell to under two days in June, according to port authority data.'),
    ('N06', '2026-05-03', 'Kestrel Steelworks wins regional manufacturing award',
     'The Sheffield cutlery and cookware maker Kestrel Steelworks won a regional award for apprenticeship training.'),
    ('N07', '2026-09-18', 'Typhoon season disrupts Kaohsiung sailings',
     'Two sailings from Kaohsiung were delayed by up to five days during September storms. Carriers expect normal schedules in October.'),
    ('N08', '2026-04-22', 'Atlas Kitchen Trading opens UK warehouse',
     'Atlas Kitchen Trading of Valencia opened a bonded warehouse in Northampton and now quotes delivered-duty-paid prices to UK '
     'retailers. The company sources part of its range from contract factories in Spain and Portugal.'),
    ('N09', '2026-03-15', 'Rupee weakens against sterling',
     'The Indian rupee fell about 3% against sterling in the first quarter, making Indian exports cheaper for UK buyers.'),
    ('N10', '2026-08-29', 'Duro Cookware expands enamel line',
     'Duro Cookware of Porto added a second enamel furnace and says lead times will stay at about two weeks.'),
    ('N11', '2026-02-10', 'Golden Wok Industrial raises minimum orders',
     'Golden Wok Industrial told distributors that its minimum order will rise to 2,000 units per product line from January.'),
    ('N12', '2026-07-08', 'Hanil Kitchen adds production shift',
     'Hanil Kitchen of Busan added a night shift but still quotes five weeks of production time for new export customers.'),
    ('N13', '2026-01-19', 'Sun Harbour Kitchen wins design prize',
     'A Sun Harbour Kitchen mixing bowl set won a packaging design prize at a Shanghai trade fair.'),
    ('N14', '2026-09-25', 'Vistula Enamel: single furnace maintenance',
     'Vistula Enamel will shut its only enamel furnace for two weeks of maintenance in February 2027.'),
    ('N15', '2026-06-30', 'Steel prices stable in Europe',
     'European flat steel prices were broadly stable in the second quarter, according to an industry survey.'),
    ('N16', '2026-05-27', 'Anatolia Metal opens second plant',
     'Anatolia Metal of Izmir opened a second pressing plant, which the company says will cut lead times.'),
]

# ---------------------------------------------------------------- filler: the long, realistic text around the few facts that matter
INTRO = ['{name} has made kitchenware in {city} since {year}.', 'The company was founded in {year} as a family workshop in {city}.',
         'Founded in {year}, {name} is one of the established cookware makers of {country}.']
PROSE = [
    'Our design team works with professional chefs to test every product in real kitchens before it reaches our catalogue.',
    'We believe good cookware should last for decades, so we design every handle, rivet and base to be repaired rather than replaced.',
    'Every pan is checked by hand for flatness and coating thickness before it is packed.',
    'Our showroom welcomes trade buyers by appointment, and samples can be sent within a week of a request.',
    'We publish an annual sustainability report and track the energy used per tonne of finished product.',
    'Our packaging is plastic-free and printed with vegetable inks; outer cartons use recycled board.',
    'Customers in more than twenty countries sell our products under their own brands or ours.',
    'We offer private-label service with custom handles, colours and gift boxes for orders above our standard minimum.',
    'Our engineers run accelerated wear tests on every coating, equal to about five years of daily home use.',
    'All steel is traceable to the mill, and every batch carries a certificate of composition.',
    'We train our own apprentices and many of our supervisors started on the factory floor.',
    'Our quality laboratory tests food-contact materials to the European framework regulation and to the relevant UK rules.',
    'Retail partners praise our reliable packing, clear barcodes and accurate shipping documents.',
    'Seasonal colours change twice a year, and our core range stays the same so that retailers can reorder with confidence.',
    'We exhibit every year at Ambiente in Frankfurt and at the main Asian homeware fairs.',
    'A dedicated export team answers trade enquiries in English within one working day.',
    'Our induction-ready bases are tested on gas, ceramic, electric and induction hobs.',
    'Handles are riveted with stainless steel rivets and tested to carry five times the weight of a full pan.',
]
QUOTES = ['"Consistent quality and honest communication," says a buyer at a European department store.',
          '"The best enamel we have tested at this price," says a cookware reviewer.',
          '"Their samples matched the production run exactly," says a homeware importer.',
          '"Reliable partner for our private-label range," says a regional retail chain.']


def page(s, rnd):
    sid, name, city, country, port, cur, price, inc, lead, moq, fac = s
    year = rnd.randint(1952, 2012)
    paras = [rnd.choice(INTRO).format(name=name, city=city, country=country, year=year) + ' ' + ' '.join(rnd.sample(PROSE, 3))]
    paras.append('## Products\n' + f'Our range covers frying pans, saucepans, chef\'s knives, mixing bowls and oven dishes. '
                 + ' '.join(rnd.sample(PROSE, 3)))
    terms = [f'Reference basket price (28 cm frying pan, 20 cm chef\'s knife, set of three mixing bowls), quoted for Northwind Home: '
             f'{price:,.2f} {cur} per basket, {inc} {"at the port of " + port if inc == "FOB" else "to the buyer\'s UK warehouse"}.',
             f'Production lead time: {lead} days from confirmed order.',
             f'Minimum order quantity: {moq:,} units per product line.',
             f'Shipping port: {port}.',
             f'Production sites: {fac} {"factory" if fac == 1 else "factories"}.',
             f'Payment terms: {rnd.choice(["30% deposit, 70% before shipping", "letter of credit", "60 days after delivery for approved accounts"])}.']
    paras.append('## Trade terms\n' + '\n'.join('- ' + t for t in terms))
    paras.append('## Quality and certification\n' + ' '.join(rnd.sample(PROSE, 3)) + ' '
                 + f'Certifications: {", ".join(rnd.sample(["ISO 9001", "ISO 14001", "BSCI", "SEDEX member", "LFGB tested", "FDA food contact"], 3))}.')
    paras.append('## Sustainability\n' + ' '.join(rnd.sample(PROSE, 4)))
    paras.append('## Our history\n' + ' '.join(rnd.sample(PROSE, 7)))
    paras.append('## What customers say\n' + ' '.join(rnd.sample(QUOTES, 2)) + ' ' + ' '.join(rnd.sample(PROSE, 2)))
    return f'# {name} ({sid})\n{city}, {country}\n\n' + '\n\n'.join(paras) + '\n'


def landed(s):
    sid, name, city, country, port, cur, price, inc, lead, moq, fac = s
    gbp = price * FX[cur]
    if inc == 'DDP':
        return round(gbp, 2)
    freight = SHIPPING[port][1]
    duty = 0.0 if (country in EU or country == 'United Kingdom') else DUTY * gbp
    return round(gbp + freight + duty, 2)


def gold():
    open_issue = {'S03': 'open safety recall (pans)', 'S19': 'suspended ethical audit'}
    rows = []
    for s in SUP:
        sid, name, city, country, port, cur, price, inc, lead, moq, fac = s
        days = lead + SHIPPING[port][0]
        why = []
        if days > MAX_DAYS:
            why.append(f'delivery {days} days > {MAX_DAYS}')
        if moq > MAX_MOQ:
            why.append(f'MOQ {moq:,} > {MAX_MOQ:,}')
        if sid in open_issue:
            why.append(open_issue[sid])
        rows.append(dict(id=sid, name=name, country=country, landed_gbp=landed(s), delivery_days=days, moq=moq, eligible=not why, reasons=why))
    ok = sorted([r for r in rows if r['eligible']], key=lambda r: r['landed_gbp'])
    return dict(picks=[r['id'] for r in ok[:3]], table=sorted(rows, key=lambda r: r['landed_gbp']),
                risks={'S09': 'closed knife recall (2025)', 'S12': 'DDP trader, contract factories', 'S04': 'single furnace, maintenance Feb 2027',
                       'S08': 'name similar to a US company with an open kettle recall'})


if __name__ == '__main__':
    rnd = random.Random(5)
    os.makedirs(os.path.join(OUT, 'suppliers'), exist_ok=True)
    open(os.path.join(OUT, 'requirements.md'), 'w').write(REQUIREMENTS)
    words = 0
    for s in SUP:
        p = page(s, rnd)
        words += len(p.split())
        open(os.path.join(OUT, 'suppliers', s[0] + '.md'), 'w').write(p)
    json.dump([dict(id=i, date=d, title=t, text=x) for i, d, t, x in NEWS], open(os.path.join(OUT, 'news.json'), 'w'), indent=1)
    json.dump({p: dict(transit_days=d, freight_gbp_per_basket=f) for p, (d, f) in SHIPPING.items()}, open(os.path.join(OUT, 'shipping.json'), 'w'), indent=1)
    json.dump([dict(id=s[0], name=s[1], city=s[2], country=s[3]) for s in SUP], open(os.path.join(OUT, 'index.json'), 'w'), indent=1)
    G = gold()
    json.dump(G, open(os.path.join(OUT, 'gold.json'), 'w'), indent=1)
    log(f'Wrote {len(SUP)} supplier pages ({words:,} words, about {words // len(SUP)} each), {len(NEWS)} news items, {len(SHIPPING)} ports.')
    log('')
    log('The known answer, computed in code from the same facts (sorted by landed cost per basket, GBP):')
    log(f'{"id":4} {"name":24} {"landed":>7} {"days":>5} {"MOQ":>6}  verdict')
    for r in G['table']:
        log(f'{r["id"]:4} {r["name"]:24} {r["landed_gbp"]:7.2f} {r["delivery_days"]:5d} {r["moq"]:6,}  '
            + ('ok' if r['eligible'] else 'REJECT: ' + '; '.join(r['reasons'])))
    log('')
    log(f'Gold picks (three cheapest that pass every rule): {", ".join(G["picks"])}')
