"""Every computed number in "How Ray-Ban Display Glasses Put a Screen in a Lens".

Inputs are either published specs (with the source in a comment) or clearly labelled example values used only to
teach a formula. Nothing here is measured on a device: it is arithmetic on public numbers.
Run: python numbers.py  ->  results/numbers_stdout.txt and results/numbers.json"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}
LOG = []


def say(s=''):
    print(s)
    LOG.append(s)


def head(s):
    say()
    say('=' * 78)
    say(s)
    say('=' * 78)


def keep(k, v):
    OUT[k] = v
    return v


deg, rad = math.degrees, math.radians

# ------------------------------------------------------------------------------------------------
# Published inputs
# ------------------------------------------------------------------------------------------------
PX = 600              # Meta: 600 x 600 pixels (Road to VR / UploadVR launch reports, iFixit: 600x600 LCoS)
FOV_DIAG = 20.0       # Meta: 20 degree field of view (diagonal per UploadVR follow-up and Guttag)
PPD_META = 42         # Meta blog: "42 pixels into each degree"
NITS_MAX = 5000       # Meta: up to 5,000 nits (Guttag confirmed about 5,000 nits)
NITS_MIN = 30         # Road to VR: 30 to 5,000 nits
HZ_DISPLAY = 90       # Road to VR: 90 Hz display, content at 30 Hz
HZ_CONTENT = 30
LEAK_META = 0.02      # Meta blog: 2% light leakage
LEAK_GUTTAG = 0.015   # Guttag: eye glow about 1.5% of the brightness reaching the eye
LUMEN_PROJ = 1.0      # Guttag (Radu's measurement): about 1 lumen out of the projector at full brightness
PITCH_UM = 3.8        # OmniVision OP03011 product brief: 3.8 um pixel pitch
NATIVE_PX = 648       # OP03011 native resolution 648 x 648; active area 2.462 mm
BATTERY_MWH = 960     # iFixit teardown: 960 mWh glasses battery
HOURS_MIXED = 6       # Meta: up to six hours mixed use
SKY_AVG = 8000        # Wikipedia, orders of magnitude (luminance): average clear sky 8 kcd/m^2
SKY_CLOUDY = 2000     # average cloudy sky 2 kcd/m^2
MONITOR = 250         # peak luminance of a typical LCD monitor 250 cd/m^2
CAT3 = (0.08, 0.18)   # EN sunglass category 3: 8 to 18% transmission
PUPIL_BRIGHT = (1.5, 4.0)   # Wikipedia, Pupil: 1.5 to 4 mm when narrow (bright light)
EMG_HZ = 2000         # Nature 2025: 2 kHz sample rate
EMG_CH = 16           # Nature 2025: 48 electrode pins in 16 bipolar channels
RX_D = 4.0            # Meta (via UploadVR): prescriptions from -4 to +4 dioptres

# ------------------------------------------------------------------------------------------------
head('1. Why a screen 2 cm from the eye cannot be read')
for d_cm in (200, 25, 2):
    say(f'focus on an object at {d_cm:>4} cm needs {100 / d_cm:5.1f} dioptres of extra focusing power')
keep('dioptre_2cm', 100 / 2)
keep('dioptre_25cm', 100 / 25)

head('2. A lens makes a virtual image far away (thin lens, example f = 10 mm)')
f = 10.0
for do in (10.0, 9.9, 9.5, 9.0):
    inv = 1 / f - 1 / do
    di = math.inf if abs(inv) < 1e-12 else 1 / inv
    txt = 'at infinity (parallel rays)' if math.isinf(di) else f'{-di / 1000:.2f} m in front of the eye (virtual image)'
    say(f'display {do:4.1f} mm from a 10 mm lens: image {txt}')
    if do == 9.9:
        keep('virtual_9p9', round(-di / 1000, 2))
    if do == 9.5:
        keep('virtual_9p5', round(-di / 1000, 2))

head("3. Snell's law: air to glass (n = 1.5) at 30 degrees")
n_air, n_glass = 1.0, 1.5
t1 = 30.0
t2 = deg(math.asin(n_air * math.sin(rad(t1)) / n_glass))
say(f'sin(30) = {math.sin(rad(t1)):.3f}; sin(theta2) = 0.5 / 1.5 = {math.sin(rad(t1)) / 1.5:.3f}; theta2 = {t2:.1f} deg')
keep('snell_30_out', round(t2, 1))
# The other way: glass to air at 30 degrees
t_out = deg(math.asin(1.5 * math.sin(rad(30))))
say(f'glass to air at 30 deg: sin(theta2) = 1.5 x 0.5 = 0.75, theta2 = {t_out:.1f} deg (bends away from the normal)')
keep('snell_glass_air_30', round(t_out, 1))
say(f'glass to air at 45 deg: 1.5 x sin(45) = {1.5 * math.sin(rad(45)):.3f} > 1, so no ray can leave: total internal reflection')
keep('sin45x15', round(1.5 * math.sin(rad(45)), 3))

head('4. Critical angle for total internal reflection, against air')
mats = [('water', 1.333), ('common glass / acrylic', 1.5), ('best glass Meta had in 2016', 1.8),
        ('lithium niobate', 2.3), ('silicon carbide (Orion)', 2.7)]
crit = {}
for name, n in mats:
    c = deg(math.asin(1 / n))
    crit[name] = round(c, 1)
    say(f'{name:<30} n = {n:<5}  critical angle = {c:4.1f} deg   TIR window = {90 - c:4.1f} deg')
keep('critical', crit)

head('5. Fresnel reflection at normal incidence (why bare glass reflects about 4%)')
R = ((1.5 - 1) / (1.5 + 1)) ** 2
say(f'R = ((1.5 - 1)/(1.5 + 1))^2 = {R:.3f} = {100 * R:.0f}%')
keep('fresnel', round(100 * R))

head('6. Zigzag inside a plate: distance between bounces on the same face (example thickness 1 mm)')
t_mm = 1.0
for a in (45, 50, 60, 70):
    say(f'ray at {a} deg from the normal in a {t_mm} mm plate: hits the same face every 2 t tan(a) = {2 * t_mm * math.tan(rad(a)):.2f} mm')
keep('bounce_50', round(2 * t_mm * math.tan(rad(50)), 2))
keep('bounce_60', round(2 * t_mm * math.tan(rad(60)), 2))
L_lens = 30.0
say(f'over an example 30 mm path at 60 deg: {L_lens / (2 * t_mm * math.tan(rad(60))):.1f} round trips')
keep('trips_30mm', round(L_lens / (2 * t_mm * math.tan(rad(60))), 1))

head('7. Angles of the picture inside the glass')
half = FOV_DIAG / 2
inside = deg(math.asin(math.sin(rad(half)) / 1.5))
say(f'a ray 10 deg off-axis in air is {inside:.2f} deg off-axis inside n = 1.5 glass')
keep('half_fov_inside', round(inside, 2))

head('8. Partly reflecting mirrors that share the light equally (US 6,829,095 examples)')
def ladder(rs):
    left, outs = 1.0, []
    for r in rs:
        outs.append(left * r)
        left *= (1 - r)
    return outs, left
for rs in ([1 / 3, 1 / 2, 1.0], [0.20, 0.25, 1 / 3]):
    outs, left = ladder(rs)
    say('reflectances ' + ', '.join(f'{100 * r:.0f}%' for r in rs) + ' -> each mirror sends ' +
        ', '.join(f'{100 * o:.0f}%' for o in outs) + f' to the eye; {100 * left:.0f}% goes on past the last mirror')
keep('ladder_20_25_33_left', round(100 * ladder([0.20, 0.25, 1 / 3])[1]))
for N in (6,):
    rs = [1 / (N - k) for k in range(N)]
    say(f'{N} mirrors, each sending 1/{N}: reflectances ' + ', '.join(f'{100 * r:.0f}%' for r in rs))

head('9. Pixels per degree')
diag_px = PX * math.sqrt(2)
side_deg = FOV_DIAG / math.sqrt(2)
ppd = diag_px / FOV_DIAG
say(f'diagonal pixels = 600 x sqrt(2) = {diag_px:.1f}')
say(f'side angle (small-angle) = 20 / sqrt(2) = {side_deg:.2f} deg')
say(f'pixels per degree = {diag_px:.1f} / 20 = {ppd:.1f}  (Meta says 42)')
say(f'one pixel spans 60 / {ppd:.1f} = {60 / ppd:.2f} arcminutes; 20/20 vision resolves about 1 arcminute = 60 ppd')
keep('diag_px', round(diag_px, 1)); keep('side_deg', round(side_deg, 2)); keep('ppd', round(ppd, 1))
keep('arcmin_per_px', round(60 / ppd, 2))
px_for_60 = 60 * side_deg
say(f'to reach 60 ppd over the same {side_deg:.2f} deg square you would need {px_for_60:.0f} x {px_for_60:.0f} pixels')
keep('px_for_60', round(px_for_60))
eff = 400 / side_deg
say(f'with Guttag\'s estimate of about 400 x 400 effective pixels: {eff:.1f} effective pixels per degree')
keep('ppd_eff_400', round(eff, 1))
# Orion-style 70 deg diagonal at 42 ppd
for fov in (30, 70):
    s = fov / math.sqrt(2)
    say(f'a square {fov} deg diagonal field at 42 ppd needs about {42 * s:.0f} x {42 * s:.0f} pixels')
    keep(f'px_{fov}_at42', round(42 * s))
    say(f'  that is {(42 * s) ** 2 / PX ** 2:.1f} times the 600 x 600 = {PX * PX:,} pixels of the Ray-Ban Display')

head('10. How big the picture looks (example: a square at a 40 cm reading distance)')
w40 = 2 * 40 * math.tan(rad(side_deg / 2))
say(f'a {side_deg:.2f} deg square is the size of a {w40:.1f} cm square seen from 40 cm')
keep('w40', round(w40, 1))

head('11. The LCoS chip and the projector focal length')
w_used = PX * PITCH_UM / 1000
w_native = NATIVE_PX * PITCH_UM / 1000
say(f'600 pixels x 3.8 um = {w_used:.2f} mm; 648 x 3.8 um = {w_native:.3f} mm (brief: 2.462 mm)')
f_mm = (w_used / 2) / math.tan(rad(side_deg / 2))
say(f'a single ideal lens that spreads {w_used:.2f} mm over {side_deg:.2f} deg needs f = {f_mm:.1f} mm')
keep('chip_mm', round(w_used, 2)); keep('f_mm', round(f_mm, 1))
say(f'pixels on the chip: {PX * PX:,}; pixel area {PITCH_UM ** 2:.1f} um^2')
keep('pixels_total', PX * PX)

head('12. Field-sequential colour')
fields = HZ_DISPLAY * 3
say(f'90 frames/s x 3 colour fields = {fields} colour fields/s, so each field lasts at most {1000 / fields:.1f} ms')
say(f'one 90 Hz frame lasts {1000 / HZ_DISPLAY:.1f} ms; one 30 Hz content update lasts {1000 / HZ_CONTENT:.1f} ms '
    f'= {HZ_DISPLAY // HZ_CONTENT} display frames')
keep('fields', fields); keep('field_ms', round(1000 / fields, 1)); keep('frame_ms', round(1000 / HZ_DISPLAY, 1))

head('13. Contrast of a see-through display against the world')
def contrast(Ld, Lbg, T):
    return (Ld + T * Lbg) / (T * Lbg)
cases = [('5,000 nits on clear sky, clear lens', NITS_MAX, SKY_AVG, 1.0),
         ('5,000 nits on clear sky, lens at 18%', NITS_MAX, SKY_AVG, CAT3[1]),
         ('5,000 nits on clear sky, lens at 8%', NITS_MAX, SKY_AVG, CAT3[0]),
         ('250 nits (monitor level) on clear sky, clear lens', MONITOR, SKY_AVG, 1.0),
         ('5,000 nits on cloudy sky, clear lens', NITS_MAX, SKY_CLOUDY, 1.0)]
cr = {}
for name, Ld, Lb, T in cases:
    c = contrast(Ld, Lb, T)
    cr[name] = round(c, 2)
    say(f'{name:<52} background {T * Lb:>7.0f} nits -> contrast {c:6.2f} : 1')
keep('contrast', cr)

head('14. Where the projector light goes: how much enters the pupil')
a = rad(side_deg)
omega = 4 * math.asin(math.sin(a / 2) * math.sin(a / 2))
say(f'solid angle of a {side_deg:.2f} x {side_deg:.2f} deg square = {omega:.4f} sr')
keep('omega', round(omega, 4))
for dp in (PUPIL_BRIGHT[0], 3.0, PUPIL_BRIGHT[1]):
    A = math.pi * (dp / 2000) ** 2
    phi = NITS_MAX * A * omega
    say(f'pupil {dp} mm: area {A * 1e6:.2f} mm^2, flux into the eye = L x A x Omega = {phi * 1000:.2f} mlm '
        f'= {100 * phi / LUMEN_PROJ:.2f}% of the 1 lm from the projector')
    if dp == 3.0:
        keep('pupil3_mlm', round(phi * 1000, 2)); keep('pupil3_pct', round(100 * phi / LUMEN_PROJ, 2))
    if dp == PUPIL_BRIGHT[1]:
        keep('pupil4_pct', round(100 * phi / LUMEN_PROJ, 2))
    if dp == PUPIL_BRIGHT[0]:
        keep('pupil15_pct', round(100 * phi / LUMEN_PROJ, 3))
# what eyebox area would carry all of it (ideal, lossless): A_box = Phi / (L Omega)
A_box = LUMEN_PROJ / (NITS_MAX * omega)
side_box = math.sqrt(A_box) * 1000
say(f'if no light were lost, 1 lm at 5,000 nits over that cone would fill {A_box * 1e6:.0f} mm^2 '
    f'(a {side_box:.0f} x {side_box:.0f} mm square)')
keep('ideal_box_mm2', round(A_box * 1e6)); keep('ideal_box_side', round(side_box))

head('15. Light that leaks forward')
for name, f_ in (('Meta, 2%', LEAK_META), ('Guttag, 1.5%', LEAK_GUTTAG)):
    say(f'{name}: at full brightness {f_ * NITS_MAX:.0f} nits leave the front, against {NITS_MAX} nits to the eye')
keep('leak_meta_nits', LEAK_META * NITS_MAX); keep('leak_guttag_nits', LEAK_GUTTAG * NITS_MAX)
say(f'brightness range: 5,000 / 30 = {NITS_MAX / NITS_MIN:.0f} times')
keep('range_x', round(NITS_MAX / NITS_MIN))

head('16. Power budget')
p = BATTERY_MWH / HOURS_MIXED
say(f'960 mWh / 6 h = {p:.0f} mW average over a day of mixed use')
keep('avg_mw', round(p))
say(f'the same battery would last {BATTERY_MWH / 1000:.1f} h at a constant 1 W')
keep('h_at_1w', round(BATTERY_MWH / 1000, 2))

head('17. Prescription range in focal lengths')
say(f'+/-{RX_D} dioptres = a lens with focal length 1/{RX_D} m = {100 / RX_D:.0f} cm')
keep('rx_focal_cm', round(100 / RX_D))

head('18. Diffraction grating: the colours split (example period 400 nm, glass n = 1.5)')
period = 400.0
grat = {}
for colour, lam in (('blue', 450), ('green', 530), ('red', 630)):
    s = lam / (1.5 * period)
    if s < 1:
        a_ = deg(math.asin(s))
        tir = a_ > crit['common glass / acrylic']
        say(f'{colour:<5} {lam} nm: sin(theta) = {lam}/(1.5 x {period:.0f}) = {s:.3f}, theta = {a_:.1f} deg, '
            f'{"trapped by TIR" if tir else "escapes"}')
        grat[colour] = round(a_, 1)
    else:
        say(f'{colour:<5} {lam} nm: sin(theta) = {s:.3f} > 1, no first-order ray inside the glass at all')
        grat[colour] = None
keep('grating', grat)

head('19. An ideal ceiling on field of view for a grating-coupled guide (one direction)')
for name, n in mats[1:]:
    x = (n - 1) / 2
    fov = 2 * deg(math.asin(min(1, x)))
    say(f'{name:<30} n = {n}: FOV <= 2 asin((n - 1)/2) = {fov:5.1f} deg (grazing rays allowed; real designs get less)')
    OUT.setdefault('fov_ceiling', {})[name] = round(fov, 1)

head('20. The Neural Band research device (Nature 2025)')
say(f'{EMG_CH} channels x {EMG_HZ} samples/s = {EMG_CH * EMG_HZ:,} numbers per second')
keep('emg_rate', EMG_CH * EMG_HZ)
say(f'handwriting: 20.9 WPM with sEMG vs 25.1 WPM writing on a surface: {20.9 / 25.1 * 100:.0f}%')
keep('wpm_ratio', round(20.9 / 25.1 * 100))

open(os.path.join(HERE, 'results', 'numbers_stdout.txt'), 'w').write('\n'.join(LOG) + '\n')
json.dump(OUT, open(os.path.join(HERE, 'results', 'numbers.json'), 'w'), indent=1)
