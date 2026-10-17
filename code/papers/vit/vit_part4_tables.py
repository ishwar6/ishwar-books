"""Part 4: the arithmetic behind the result tables of the ViT paper (no model is run here).

Every input number is copied from the paper: Table 2 (comparison with the state of the art), Table 5 (Appendix C,
the numbers behind Figure 3), Table 7 (Appendix D.1, Adam versus SGD for ResNets), Table 9 (Appendix D.10, the 19
VTAB tasks) and Appendix D.9 (ObjectNet). The script recomputes what the text says about them:
  1. TPUv3-core-days: what the unit means, the 8-core, 30-day check, and the compute ratios between columns;
  2. Table 2 as error rates and relative error reductions, and the gaps measured in standard deviations;
  3. which rows ViT-L/16 (JFT) actually wins against BiT-L;
  4. Table 2 against Table 5: what Polyak averaging and 512/518 px fine-tuning add;
  5. Table 5: the order of the five ViT sizes on each pre-training dataset (Figure 3);
  6. Table 9: the 19 VTAB tasks averaged into the three groups of Figure 2, and the means checked;
  7. Table 7: the averages checked, and Adam minus SGD;
  8. ObjectNet against ImageNet, and against BiT-L (the BiT paper's Table 6).
Writes results/part4_tables.json and results/part4_tables_stdout.txt."""
from common import Log, save

log = Log('part4_tables')
R = {}

# ---------------------------------------------------------------- Table 2 (copied from the paper)
COLS = ['ViT-H/14 (JFT)', 'ViT-L/16 (JFT)', 'ViT-L/16 (I21k)', 'BiT-L (R152x4)', 'Noisy Student (EffNet-L2)']
T2 = {   # mean, std over three fine-tuning runs; None = not reported
    'ImageNet':          [(88.55, .04), (87.76, .03), (85.30, .02), (87.54, .02), (88.4, None)],
    'ImageNet ReaL':     [(90.72, .05), (90.54, .03), (88.62, .05), (90.54, None), (90.55, None)],
    'CIFAR-10':          [(99.50, .06), (99.42, .03), (99.15, .03), (99.37, .06), None],
    'CIFAR-100':         [(94.55, .04), (93.90, .05), (93.25, .05), (93.51, .08), None],
    'Oxford-IIIT Pets':  [(97.56, .03), (97.32, .11), (94.67, .15), (96.62, .23), None],
    'Oxford Flowers-102': [(99.68, .02), (99.74, .00), (99.61, .02), (99.63, .03), None],
    'VTAB (19 tasks)':   [(77.63, .23), (76.28, .46), (72.72, .21), (76.29, 1.70), None],
}
CORE_DAYS = [2.5e3, 0.68e3, 0.23e3, 9.9e3, 12.3e3]

log('== 1. TPUv3-core-days: cores x days ==')
log('  one TPUv3 chip has 2 cores; "core-days" = (number of cores) x (days of training)')
log(f'  the paper: ViT-L/16 on ImageNet-21k "could be trained using a standard cloud TPUv3 with 8 cores in approximately 30 days"')
log(f'  8 cores x 30 days = {8 * 30} core-days = {8 * 30 / 1000:.2f}k   (Table 2 says 0.23k for ViT-L/16, I21k)')
for c, d in zip(COLS, CORE_DAYS):
    log(f'  {c:<27} {d / 1000:>5.2f}k core-days = {d / 2:>6,.0f} chip-days = {d / 8:>5,.0f} days on one 8-core TPUv3')
r_bit_l16 = CORE_DAYS[3] / CORE_DAYS[1]
r_bit_h14 = CORE_DAYS[3] / CORE_DAYS[0]
r_ns_h14 = CORE_DAYS[4] / CORE_DAYS[0]
r_h14_l16 = CORE_DAYS[0] / CORE_DAYS[1]
log(f'  BiT-L / ViT-L/16 (JFT) = 9.9k / 0.68k = {r_bit_l16:.1f}x more pre-training compute for BiT-L')
log(f'  BiT-L / ViT-H/14       = 9.9k / 2.5k  = {r_bit_h14:.1f}x')
log(f'  Noisy Student / ViT-H/14 = 12.3k / 2.5k = {r_ns_h14:.1f}x')
log(f'  ViT-H/14 / ViT-L/16 (JFT) = 2.5k / 0.68k = {r_h14_l16:.1f}x')
R['core_days'] = dict(eight_cores_30_days=240, bit_over_l16=round(r_bit_l16, 2), bit_over_h14=round(r_bit_h14, 2),
                      ns_over_h14=round(r_ns_h14, 2), h14_over_l16=round(r_h14_l16, 2))

log('')
log('== 2. Table 2 as error rates (100 - accuracy) ==')
log(f'  {"dataset":<19}' + ''.join(f'{c.split(" (")[0]:>16}' for c in COLS[:4]) + '   H/14 vs BiT-L: error cut')
R['errors'] = {}
for k, row in T2.items():
    err = [None if v is None else 100 - v[0] for v in row]
    cut = (err[3] - err[0]) / err[3]
    R['errors'][k] = dict(h14=round(err[0], 2), l16=round(err[1], 2), bit=round(err[3], 2), cut=round(cut, 3))
    log(f'  {k:<19}' + ''.join(f'{e:>16.2f}' for e in err[:4]) + f'   {err[3]:.2f} -> {err[0]:.2f}: {100 * cut:.0f}% fewer errors')

log('')
log('== 3. ViT-L/16 (JFT) against BiT-L, row by row (gap in points, and in units of the larger std) ==')
wins = 0
R['l16_vs_bit'] = {}
for k, row in T2.items():
    (a, sa), (b, sb) = row[1], row[3]
    gap = a - b
    s = max(sa or 0, sb or 0)
    verdict = 'ViT-L/16 higher' if gap > 0 else ('equal' if gap == 0 else 'BiT-L higher')
    wins += gap > 0
    R['l16_vs_bit'][k] = dict(gap=round(gap, 2), std=s, verdict=verdict)
    log(f'  {k:<19} {a:6.2f} - {b:6.2f} = {gap:+.2f}   (larger std {s:.2f}{", gap = %.1f std" % (gap / s) if s else ""})   {verdict}')
log(f'  ViT-L/16 (JFT) is strictly higher on {wins} of 7 rows; ImageNet ReaL is a tie and VTAB is 0.01 lower, inside BiT-L\'s std of 1.70')
R['l16_wins'] = wins

log('  ViT-H/14 minus ViT-L/16 (both JFT), row by row: ' + ', '.join(f'{k} {row[0][0] - row[1][0]:+.2f}' for k, row in T2.items()))
log(f'  ViT-L/16 (I21k) against BiT-L on ImageNet: {85.30 - 87.54:+.2f} points, for 0.23k against 9.9k core-days')
R['h14_minus_l16'] = {k: round(row[0][0] - row[1][0], 2) for k, row in T2.items()}

log('')
log('== 4. Table 2 versus Table 5: ImageNet with and without Polyak averaging and 512/518 px fine-tuning ==')
pairs = [('ViT-H/14 (JFT)', 88.55, 88.04), ('ViT-L/16 (JFT)', 87.76, 87.12), ('ViT-L/16 (I21k)', 85.30, 85.15)]
R['polyak'] = {}
for n, t2, t5 in pairs:
    R['polyak'][n] = round(t2 - t5, 2)
    log(f'  {n:<16} Table 2: {t2:.2f}   Table 5 (384 px, no averaging): {t5:.2f}   difference {t2 - t5:+.2f}')

log('')
log('== 5. Table 5: ImageNet top-1 after fine-tuning, by pre-training dataset (the numbers behind Figure 3) ==')
SIZES = ['ViT-B/16', 'ViT-B/32', 'ViT-L/16', 'ViT-L/32', 'ViT-H/14']
T5 = {   # pre-training dataset -> fine-tuned accuracy per model on ImageNet, CIFAR-100, Pets
    'ImageNet':     {'ImageNet': [77.91, 73.38, 76.53, 71.16, None], 'CIFAR-100': [87.13, 86.31, 86.35, 87.07, None],
                     'Pets': [93.81, 92.04, 93.64, 91.35, None]},
    'ImageNet-21k': {'ImageNet': [83.97, 81.28, 85.15, 80.99, 85.13], 'CIFAR-100': [91.67, 91.97, 93.44, 93.04, 93.82],
                     'Pets': [94.43, 93.02, 94.73, 93.09, 94.82]},
    'JFT-300M':     {'ImageNet': [84.15, 80.73, 87.12, 84.37, 88.04], 'CIFAR-100': [91.87, 90.49, 94.04, 92.52, 94.55],
                     'Pets': [95.80, 93.40, 97.11, 95.83, 97.56]},
}
R['table5'] = {}
for ds, rows in T5.items():
    im = rows['ImageNet']
    order = sorted([(v, n) for n, v in zip(SIZES, im) if v is not None], reverse=True)
    lb = im[2] - im[0]
    R['table5'][ds] = dict(imagenet=im, order=[n for _, n in order], l16_minus_b16=round(lb, 2))
    log(f'  pre-trained on {ds:<13} ' + '  '.join(f'{n} {v:.2f}' for n, v in zip(SIZES, im) if v is not None))
    log(f'      best to worst: {" > ".join(n for _, n in order)};   ViT-L/16 - ViT-B/16 = {lb:+.2f}')
g_b16 = T5['JFT-300M']['ImageNet'][0] - T5['ImageNet']['ImageNet'][0]
g_l16 = T5['JFT-300M']['ImageNet'][2] - T5['ImageNet']['ImageNet'][2]
log(f'  ImageNet -> JFT-300M pre-training: ViT-B/16 gains {g_b16:+.2f} points, ViT-L/16 gains {g_l16:+.2f} points')
R['table5_gain'] = dict(b16=round(g_b16, 2), l16=round(g_l16, 2))

log('')
log('== 6. Table 9: the 19 VTAB-1k tasks, grouped as in Figure 2 ==')
TASKS = ['Caltech101', 'CIFAR-100', 'DTD', 'Flowers102', 'Pets', 'Sun397', 'SVHN',
         'Camelyon', 'EuroSAT', 'Resisc45', 'Retinopathy',
         'Clevr-Count', 'Clevr-Dist', 'DMLab', 'dSpr-Loc', 'dSpr-Ori', 'KITTI-Dist', 'sNORB-Azim', 'sNORB-Elev']
GROUPS = {'Natural': TASKS[:7], 'Specialized': TASKS[7:11], 'Structured': TASKS[11:]}
T9 = {
    'ViT-H/14 (JFT)': ([95.3, 85.5, 75.2, 99.7, 97.2, 65.0, 88.9, 83.3, 96.7, 91.4, 76.6, 91.7, 63.8, 53.1, 79.4, 63.3, 84.5, 33.2, 51.2], 77.6),
    'ViT-L/16 (JFT)': ([95.4, 81.9, 74.3, 99.7, 96.7, 63.5, 87.4, 83.6, 96.5, 89.7, 77.1, 86.4, 63.1, 49.7, 74.5, 60.5, 82.2, 36.2, 51.1], 76.3),
    'ViT-L/16 (I21k)': ([90.8, 84.1, 74.1, 99.3, 92.7, 61.0, 80.9, 82.5, 95.6, 85.2, 75.3, 70.3, 56.1, 41.9, 74.7, 64.9, 79.9, 30.5, 41.7], 72.7),
}
R['vtab'] = {}
for name, (vals, mean) in T9.items():
    d = dict(zip(TASKS, vals))
    m = sum(vals) / len(vals)
    gm = {g: sum(d[t] for t in ts) / len(ts) for g, ts in GROUPS.items()}
    lo = min(d, key=d.get)
    R['vtab'][name] = dict(mean=round(m, 2), table_mean=mean, groups={g: round(v, 2) for g, v in gm.items()}, lowest=[lo, d[lo]])
    log(f'  {name:<16} mean of 19 = {m:.2f} (table: {mean})   Natural {gm["Natural"]:.1f}   Specialized {gm["Specialized"]:.1f}   '
        f'Structured {gm["Structured"]:.1f}   lowest task: {lo} {d[lo]}')
h, l = T9['ViT-H/14 (JFT)'][0], T9['ViT-L/16 (I21k)'][0]
diff = sorted(((a - b, t) for t, a, b in zip(TASKS, h, l)), reverse=True)
log(f'  ViT-H/14 (JFT) minus ViT-L/16 (I21k), largest gaps: ' + ', '.join(f'{t} {g:+.1f}' for g, t in diff[:4]))
log(f'  ... smallest: ' + ', '.join(f'{t} {g:+.1f}' for g, t in diff[-3:]))
R['vtab_gap_top'] = [[t, round(g, 1)] for g, t in diff[:4]]
R['vtab_gap_bottom'] = [[t, round(g, 1)] for g, t in diff[-3:]]

log('')
log('== 7. Table 7 (Appendix D.1): ResNets pre-trained on JFT with Adam or SGD, then fine-tuned ==')
T7 = {'ImageNet': (77.54, 78.24, 84.97, 84.37), 'CIFAR10': (97.67, 97.46, 99.06, 99.07), 'CIFAR100': (86.07, 85.17, 92.05, 91.06),
      'Oxford-IIIT Pets': (91.11, 91.00, 95.37, 94.79), 'Oxford Flowers-102': (94.26, 92.06, 98.62, 99.32)}
avg = [sum(v[i] for v in T7.values()) / 5 for i in range(4)]
log(f'  {"":<19}{"R50 Adam":>10}{"R50 SGD":>10}{"Adam-SGD":>10}{"R152x2 Adam":>13}{"R152x2 SGD":>12}{"Adam-SGD":>10}')
adam_wins = 0
for k, v in T7.items():
    adam_wins += (v[0] > v[1]) + (v[2] > v[3])
    log(f'  {k:<19}{v[0]:>10.2f}{v[1]:>10.2f}{v[0] - v[1]:>+10.2f}{v[2]:>13.2f}{v[3]:>12.2f}{v[2] - v[3]:>+10.2f}')
log(f'  {"average (ours)":<19}{avg[0]:>10.2f}{avg[1]:>10.2f}{avg[0] - avg[1]:>+10.2f}{avg[2]:>13.2f}{avg[3]:>12.2f}{avg[2] - avg[3]:>+10.2f}')
log(f'  {"average (table)":<19}{89.33:>10.2f}{88.79:>10.2f}{"":>10}{94.01:>13.2f}{93.72:>12.2f}')
log(f'  Adam is higher in {adam_wins} of 10 dataset-model pairs')
R['table7'] = dict(avg=[round(a, 2) for a in avg], adam_minus_sgd=[round(avg[0] - avg[1], 2), round(avg[2] - avg[3], 2)], adam_wins=adam_wins)

log('')
log('== 8. ObjectNet (Appendix D.9) ==')
log(f'  ViT-H/14: ObjectNet top-1 61.7%, top-5 82.1%;  ImageNet top-1 (Table 2) 88.55%')
log(f'  drop from ImageNet to ObjectNet, top-1: {88.55 - 61.7:.2f} points')
log(f'  BiT-L (R152x4, JFT), the BiT paper, Table 6, resize and crop: ObjectNet top-1 58.7%, top-5 80.0%')
log(f'  ViT-H/14 minus BiT-L on ObjectNet: top-1 {61.7 - 58.7:+.1f}, top-5 {82.1 - 80.0:+.1f} points')
log(f'  BiT paper, Table 6, top-1, BiT-S -> BiT-L: R50x1 30.8 -> 37.6 ({37.6 - 30.8:+.1f}), R152x4 36.0 -> 58.7 ({58.7 - 36.0:+.1f})')
R['objectnet'] = dict(top1=61.7, top5=82.1, drop=round(88.55 - 61.7, 2), bit_top1=58.7, bit_top5=80.0)

save('part4_tables', R)
