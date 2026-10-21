"""SVG figures for Parts 3, 4 and 5 of "How AI Got Here" (reinforcement learning; how language models are trained today;
inside a modern language model).
CSS classes only (box-1..4, box-on, edge, arrow, t-title, t-note, t-tick, t-muted, l1, l2, axis, grid...), never colours,
so every figure follows the site theme. The numbers in the charts come from rl_history.py (results/rl_history_stdout.txt)
and, for Part 5, pretrain_history.py (results/pretrain_history.json).
Run: python figs_rl.py  ->  results/figs_rl.json  (used by assemble_rl.py)"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
F = {}


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def text(x, y, s, cls='t-note', anchor=None):
    a = f' text-anchor="{anchor}"' if anchor else ''
    return f'<text class="{cls}" x="{x:g}" y="{y:g}"{a}>{esc(s)}</text>'


def box(x, y, w, h, cls='box', lines=(), sub_cls='t-muted', rx=6):
    out = [f'<rect class="{cls}" x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx}"/>']
    lines = list(lines)
    n = len(lines)
    if n:
        gap = 17
        y0 = y + h / 2 - (n - 1) * gap / 2 + 4.5
        for i, s in enumerate(lines):
            out.append(text(x + w / 2, y0 + i * gap, s, 't-note' if i == 0 else sub_cls, 'middle'))
    return ''.join(out)


def head(x, y, dx, dy, cls='arrow', size=8):
    L = math.hypot(dx, dy) or 1
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    p1 = (x, y)
    p2 = (x - ux * size + px * size * 0.5, y - uy * size + py * size * 0.5)
    p3 = (x - ux * size - px * size * 0.5, y - uy * size - py * size * 0.5)
    return f'<polygon class="{cls}" points="{p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f} {p3[0]:.1f},{p3[1]:.1f}"/>'


def arrow(pts, cls='edge', hcls='arrow'):
    """A polyline through pts with an arrowhead at the last point."""
    d = ' '.join(f'{x:g},{y:g}' for x, y in pts)
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    return f'<polyline class="{cls}" points="{d}"/>' + head(x2, y2, x2 - x1, y2 - y1, hcls)


def svg(key, w, h, label, body, caption):
    F[key] = (f'<figure class="fig"><svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}">'
              + ''.join(body) + f'</svg><figcaption>{caption}</figcaption></figure>')


def timeline(key, title, start, end, ticks, rows, label, caption):
    x0, x1 = 250, 700
    sx = lambda yr: x0 + (yr - start) / (end - start) * (x1 - x0)
    H = 34 + 40 * len(rows) + 50
    b = [text(10, 22, title, 't-title')]
    for t in ticks:
        b.append(f'<line class="grid" x1="{sx(t):.1f}" y1="34" x2="{sx(t):.1f}" y2="{H - 38}"/>')
    for i, (name, span, a, z, cls) in enumerate(rows):
        y = 42 + 40 * i
        b.append(text(10, y + 10, name, 't-note'))
        b.append(text(10, y + 26, span, 't-tick'))
        b.append(f'<rect class="{cls}" x="{sx(a):.1f}" y="{y}" width="{max(6, sx(z) - sx(a)):.1f}" height="22" rx="4"/>')
    b.append(f'<line class="axis" x1="{x0}" y1="{H - 36}" x2="{x1}" y2="{H - 36}"/>')
    for t in ticks:
        b.append(text(sx(t), H - 16, str(t), 't-tick', 'middle'))
    svg(key, 720, H, label, b, caption)


# =============================================================================================
# Part 3
# =============================================================================================
timeline('p3-timeline', 'Seven threads of reinforcement learning, 1890 to 2025', 1890, 2025,
         [1900, 1925, 1950, 1975, 2000, 2025],
         [('Animal learning, law of effect', '1898 to 1911', 1898, 1911, 'box-1'),
          ('Dynamic programming, MDPs', '1953 to 1960', 1953, 1960, 'box-2'),
          ('First learning machines', '1951 to 1963', 1951, 1963, 'box-3'),
          ('Bandits, exploration', '1933 to 2002', 1933, 2002, 'box-4'),
          ('TD, Q-learning, policy gradients', '1983 to 2000', 1983, 2000, 'box-1'),
          ('Deep RL from pixels', '2013 to 2017', 2013, 2017, 'box-2'),
          ('Self-play and search', '2015 to 2020', 2015, 2020, 'box-3')],
         'A timeline of seven threads of reinforcement learning, from animal experiments in 1898 to self-play systems around 2020.',
         'The ideas are old and the big results are recent. Most of the theory existed by 2000. Deep networks and fast hardware made it work at scale after 2013.')

# --- the agent and environment loop -------------------------------------------------------
b = [text(10, 22, 'The reinforcement learning loop', 't-title'),
     box(60, 90, 190, 80, 'box-on', ['Agent', 'chooses an action']),
     box(470, 90, 190, 80, 'box-2', ['Environment', 'a game, a robot, a world']),
     arrow([(250, 112), (470, 112)], 'edge-on', 'arrow-on'),
     text(360, 104, 'action  a', 't-note', 'middle'),
     arrow([(470, 150), (250, 150)], 'edge-2', 'arrow-2'),
     text(360, 172, 'new state  s′  and reward  r', 't-note', 'middle'),
     text(360, 228, 'Repeat: s, a, r, s′, a′, r′, ... The agent wants the largest total reward over time.', 't-muted', 'middle'),
     text(360, 250, 'Nobody tells it the right action. It must try actions and learn from what comes back.', 't-muted', 'middle')]
svg('p3-loop', 720, 270, 'The agent sends an action to the environment. The environment returns a new state and a reward.', b,
    'Every method in this part fits this loop. Only the way the agent learns from the rewards changes.')

# --- the corridor ------------------------------------------------------------------------
def corridor(key, title, rows, label, caption):
    b = [text(10, 22, title, 't-title')]
    names = ['0: pit', '1', '2: start', '3', '4: goal']
    cls = ['box-4', 'box', 'box-on', 'box', 'box-1']
    for i in range(5):
        x = 70 + i * 120
        b.append(box(x, 44, 100, 56, cls[i], [names[i]]))
    b.append(text(120, 120, 'reward −1', 't-muted', 'middle'))
    b.append(text(600, 120, 'reward +1', 't-muted', 'middle'))
    y = 150
    for lab, vals in rows:
        b.append(text(10, y + 4, lab, 't-note'))
        for i, v in enumerate(vals):
            if v is not None:
                b.append(text(120 + i * 120, y + 26, v, 't-big', 'middle'))
        y += 62
    svg(key, 720, y - 6, label, b, caption)


corridor('p3-corridor', 'A five-cell corridor: values for two policies (γ = 0.9)',
         [('Random policy (left or right, half and half): V', ['end', '−0.5', '0', '0.5', 'end']),
          ('Best policy (always right): V*', ['end', '0.81', '0.9', '1', 'end'])],
         'A corridor of five cells, a pit on the left with reward minus one and a goal on the right with reward plus one. '
         'Under a random policy the values of cells 1 to 3 are minus 0.5, 0 and 0.5. Under the best policy they are 0.81, 0.9 and 1.',
         'The same cells have different values under different policies. A value is "how good it is to be here, if I keep acting this way". Numbers from our script.')

# --- the backup diagram -----------------------------------------------------------------------
b = [text(10, 22, 'One Bellman backup at cell 2 (best policy, γ = 0.9)', 't-title'),
     box(290, 44, 140, 50, 'box-on', ['V*(2) = 0.9']),
     box(90, 200, 160, 54, 'box', ['cell 1', 'V*(1) = 0.81']),
     box(470, 200, 160, 54, 'box', ['cell 3', 'V*(3) = 1']),
     arrow([(330, 94), (200, 200)], 'edge', 'arrow'),
     arrow([(390, 94), (520, 200)], 'edge-on', 'arrow-on'),
     text(250, 140, 'left: 0 + 0.9 × 0.81 = 0.729', 't-note', 'end'),
     text(470, 140, 'right: 0 + 0.9 × 1 = 0.9', 't-note'),
     text(360, 292, 'V*(2) = max(0.729, 0.9) = 0.9.  The best action is "right".', 't-note', 'middle')]
svg('p3-backup', 720, 312, 'A backup diagram: the value of cell 2 is the larger of two options, left with 0.729 and right with 0.9.', b,
    'A backup looks one step ahead, adds the reward to the discounted value of the next cell, and keeps the best. Value iteration does this for every cell, again and again.')

# --- a Q-table update -------------------------------------------------------------------------
b = [text(10, 22, 'One Q-learning update (α = 0.5, γ = 0.9)', 't-title'),
     text(130, 56, 'left', 't-label', 'middle'), text(230, 56, 'right', 't-label', 'middle')]
vals = {(1, 0): '0', (1, 1): '0', (2, 0): '0', (2, 1): '0 → 0.225', (3, 0): '0', (3, 1): '0.5'}
for r, s in enumerate((1, 2, 3)):
    y = 66 + r * 44
    b.append(text(40, y + 27, f'cell {s}', 't-note'))
    for c in (0, 1):
        on = (s, c) == (2, 1)
        b.append(f'<rect class="{"box-on" if on else ("box-1" if (s, c) == (3, 1) else "box")}" x="{80 + c * 100}" y="{y}" width="100" height="40" rx="4"/>')
        b.append(text(130 + c * 100, y + 25, vals[(s, c)], 't-note', 'middle'))
b += [text(330, 76, 'The agent is in cell 2, moves right, gets reward 0, lands in cell 3.', 't-muted'),
      text(330, 104, 'target = r + γ · max Q(3, ·) = 0 + 0.9 × 0.5 = 0.45', 't-note'),
      text(330, 132, 'error  = target − Q(2, right) = 0.45 − 0 = 0.45', 't-note'),
      text(330, 160, 'new Q(2, right) = 0 + 0.5 × 0.45 = 0.225', 't-note'),
      text(330, 196, 'The value 0.5 in cell 3 came from the goal one step earlier.', 't-muted'),
      text(330, 216, 'Each update pulls reward one cell further back.', 't-muted'),
      arrow([(280, 130), (322, 130)], 'edge-on', 'arrow-on')]
svg('p3-qtable', 720, 240, 'A Q-table with three rows and two columns. The cell for state 2, action right, changes from 0 to 0.225.', b,
    'A Q-table holds one number per state and action. The update moves one entry part of the way toward "reward now plus the best value next". These are the third and fourth updates of our run.')

# --- bandit chart ----------------------------------------------------------------------------
b = [text(10, 22, 'Two slot machines, 1,000 pulls, 2,000 runs: mean reward per pull', 't-title')]
x0, x1, yb, yt = 200, 680, 200, 50
sx = lambda v: x0 + (v - 0.4) / (0.62 - 0.4) * (x1 - x0)
for t in (0.4, 0.45, 0.5, 0.55, 0.6):
    b.append(f'<line class="grid" x1="{sx(t):.1f}" y1="{yt - 6}" x2="{sx(t):.1f}" y2="{yb}"/>')
    b.append(text(sx(t), yb + 18, f'{t:.2f}', 't-tick', 'middle'))
bars = [('Greedy (ε = 0)', 0.5155, 'box-4'), ('ε-greedy (ε = 0.1)', 0.5815, 'box-2'), ('Always the best arm', 0.6, 'box-1')]
for i, (n, v, c) in enumerate(bars):
    y = yt + i * 48
    b.append(text(10, y + 20, n, 't-note'))
    b.append(f'<rect class="{c}" x="{x0}" y="{y}" width="{sx(v) - x0:.1f}" height="30" rx="4"/>')
    b.append(text(sx(v) + 8, y + 20, f'{v:.4f}', 't-val'))
b.append(f'<line class="axis" x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}"/>')
svg('p3-bandit', 720, 232, 'A bar chart. Greedy play earns 0.5155 per pull, epsilon-greedy earns 0.5815, and always pulling the best arm would earn 0.6.', b,
    'Arm A pays 1 with probability 0.4 and arm B with probability 0.6. Pure greed often locks onto the worse arm (it ends on arm B in only 57.9% of runs). A little exploration finds B every time. Numbers from our script.')

# --- DQN architecture with replay and target network ------------------------------------
b = [text(10, 22, 'DQN: a Q-network, a replay buffer and a frozen copy', 't-title'),
     box(20, 60, 130, 60, 'box-2', ['Atari screen', '4 frames, 84 × 84']),
     box(190, 50, 170, 80, 'box-on', ['Q-network', '3 conv + 2 dense layers', 'one output per action']),
     box(400, 60, 140, 60, 'box', ['pick action', 'ε-greedy on Q']),
     box(580, 60, 120, 60, 'box-2', ['game', 'emulator']),
     arrow([(150, 90), (190, 90)]), arrow([(360, 90), (400, 90)]), arrow([(540, 90), (580, 90)]),
     box(420, 190, 280, 60, 'box-3', ['replay buffer', 'the last 1 million frames of play']),
     arrow([(640, 120), (640, 190)], 'edge-2', 'arrow-2'),
     text(648, 160, 'store', 't-muted'),
     box(190, 290, 170, 60, 'box-4', ['target network', 'old copy, updated rarely']),
     box(420, 290, 280, 60, 'box', ['loss on a random minibatch', '(r + γ max Q_target(s′) − Q(s, a))²']),
     arrow([(560, 250), (560, 290)]), text(568, 276, 'sample', 't-muted'),
     arrow([(360, 320), (420, 320)]),
     arrow([(470, 350), (470, 368), (110, 368), (110, 120), (190, 110)], 'edge-on', 'arrow-on'),
     text(118, 236, 'gradient', 't-muted'),
     arrow([(275, 130), (275, 290)], 'edge-dim', 'arrow'), text(283, 214, 'copy every', 't-muted'), text(283, 230, 'C steps', 't-muted')]
svg('p3-dqn', 720, 384, 'The DQN loop: screens go into a Q-network, actions go to the game, experience goes into a replay buffer, '
    'minibatches from the buffer train the network against a frozen target copy.', b,
    'The two fixes of DQN. The replay buffer breaks the chain of similar frames by training on random old steps. The frozen target network keeps the target from moving at every step. The 1-million-frame memory is the size the 2015 Nature paper reports.')

# --- actor-critic loop ----------------------------------------------------------------------
b = [text(10, 22, 'Actor-critic: one network acts, one network judges', 't-title'),
     box(40, 60, 180, 70, 'box-on', ['actor  π(a | s)', 'the policy']),
     box(500, 60, 180, 70, 'box-2', ['environment']),
     box(270, 200, 180, 70, 'box-3', ['critic  V(s)', 'the value estimate']),
     arrow([(220, 85), (500, 85)], 'edge-on', 'arrow-on'), text(360, 77, 'action a', 't-note', 'middle'),
     arrow([(590, 130), (590, 235), (450, 235)], 'edge-2', 'arrow-2'), text(598, 190, 'r, s′', 't-note'),
     text(360, 300, 'TD error  δ = r + γ V(s′) − V(s)', 't-note', 'middle'),
     arrow([(270, 235), (130, 235), (130, 130)], 'edge-on', 'arrow-on'),
     text(30, 190, '"better than', 't-muted'), text(30, 206, 'expected by δ"', 't-muted'),
     text(360, 322, 'The actor makes the action more likely if δ > 0, less likely if δ < 0. The critic moves V(s) toward r + γ V(s′).', 't-muted', 'middle')]
svg('p3-actorcritic', 720, 340, 'The actor sends actions to the environment. The critic turns the reward into a TD error, which tells the actor whether the action was better or worse than expected.', b,
    'The critic replaces the noisy total return of REINFORCE with a one-step estimate. That lowers the variance, at the cost of some bias from the critic\'s own mistakes.')

# --- PPO clip chart -------------------------------------------------------------------------
def clip_panel(ox, A, title):
    out = [text(ox + 150, 44, title, 't-note', 'middle')]
    X0, X1, Y0, Y1 = ox + 30, ox + 290, 70, 250
    rx = lambda r: X0 + r / 2.2 * (X1 - X0)
    ry = lambda v: Y1 - (v + 2.2) / 4.4 * (Y1 - Y0)
    out.append(f'<line class="axis" x1="{X0}" y1="{ry(0):.1f}" x2="{X1}" y2="{ry(0):.1f}"/>')
    out.append(f'<line class="axis" x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}"/>')
    for r in (0.8, 1.2):
        out.append(f'<line class="grid" x1="{rx(r):.1f}" y1="{Y0}" x2="{rx(r):.1f}" y2="{Y1}"/>')
    for r in (0.5, 1.0, 1.5, 2.0):
        out.append(text(rx(r), ry(0) + (16 if A > 0 else -8), f'{r:g}', 't-tick', 'middle'))
    out.append(text(rx(0.8), Y1 + 16 if A > 0 else Y0 - 6, '0.8', 't-muted', 'middle'))
    out.append(text(rx(1.2), Y1 + 16 if A > 0 else Y0 - 6, '1.2', 't-muted', 'middle'))
    rs = [i / 100 for i in range(0, 221)]
    un = ' '.join(f'{rx(r):.1f},{ry(r * A):.1f}' for r in rs)
    cl = ' '.join(f'{rx(r):.1f},{ry(min(r * A, max(0.8, min(1.2, r)) * A)):.1f}' for r in rs)
    out.append(f'<polyline class="edge-dim" points="{un}"/>')
    out.append(f'<polyline class="key-line" points="{cl}"/>')
    for r in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0):
        v = min(r * A, max(0.8, min(1.2, r)) * A)
        out.append(f'<circle class="s1" cx="{rx(r):.1f}" cy="{ry(v):.1f}" r="4"/>')
    return out


b = [text(10, 22, 'The PPO clipped objective for one action (ε = 0.2)', 't-title')]
b += clip_panel(40, 1, 'advantage A = +1 (a good action)')
b += clip_panel(390, -1, 'advantage A = −1 (a bad action)')
b += [text(60, 290, 'Thick line: min(r·A, clip(r, 0.8, 1.2)·A). Dotted line: r·A without the clip. x-axis: the ratio r.', 't-muted'),
      text(60, 310, 'Flat parts have zero slope, so the gradient stops pushing once the policy has moved far enough.', 't-muted')]
svg('p3-ppo', 720, 326, 'Two panels. For a positive advantage the objective rises with the ratio and goes flat at 1.2. '
    'For a negative advantage the objective is flat below 0.8 and keeps falling above 1.', b,
    'The dots are the rows of our script: at r = 1.5 a good action scores 1.2, not 1.5, and a bad action scores −1.5, not −1.2. The clip limits gains, never losses.')

# --- AlphaZero loop -------------------------------------------------------------------------
b = [text(10, 22, 'AlphaZero: search makes the targets, the network learns them', 't-title'),
     box(30, 60, 200, 80, 'box-on', ['network f(s)', 'policy p and value v']),
     box(300, 60, 190, 80, 'box-3', ['tree search (MCTS)', '800 simulations a move', 'guided by p and v']),
     box(550, 60, 150, 80, 'box-2', ['self-play game', 'play the move', 'the search likes']),
     arrow([(230, 100), (300, 100)]), arrow([(490, 100), (550, 100)]),
     box(300, 210, 400, 70, 'box', ['training data for each position', 'search visit counts π  and the final result z (win +1, loss −1)']),
     arrow([(625, 140), (625, 210)], 'edge-2', 'arrow-2'),
     arrow([(300, 245), (130, 245), (130, 140)], 'edge-on', 'arrow-on'),
     text(30, 275, 'train p toward π', 't-note'), text(30, 293, 'and v toward z', 't-note'),
     text(360, 318, 'The search is a better player than the raw network, so the network learns to copy a stronger player: itself, after thinking.', 't-muted', 'middle')]
svg('p3-alphazero', 720, 336, 'The AlphaZero loop: the network guides a tree search, the search plays self-play games, '
    'and the visit counts and game results train the network.', b,
    'Policy improvement and policy evaluation, the two halves of Bellman\'s dynamic programming, here done by a search and a network. 800 simulations per move is the setting reported for AlphaZero.')

# =============================================================================================
# Part 4
# =============================================================================================
timeline('p4-timeline', 'From preferences to reasoning, 2017 to 2026', 2016.5, 2026.9,
         [2017, 2019, 2021, 2023, 2025],
         [('Learning from preferences', '2017 to 2020', 2017.45, 2020.7, 'box-1'),
          ('RLHF assistants', '2022 to 2023', 2022.2, 2023.6, 'box-2'),
          ('Constitutional AI, RLAIF', '2022 to 2023', 2022.95, 2023.7, 'box-3'),
          ('DPO and its family', '2023 to 2024', 2023.4, 2024.5, 'box-4'),
          ('Verifiers, process rewards', '2022 to 2024', 2022.2, 2024.9, 'box-1'),
          ('Reasoning RL: o1, R1, RLVR', '2024 to 2026', 2024.7, 2026.8, 'box-2'),
          ('RL for agents and tools', '2025 to 2026', 2025.0, 2026.8, 'box-3')],
         'A timeline of post-training methods from 2017 to 2026. Preference learning comes first, then RLHF assistants, '
         'Constitutional AI, DPO, verifiers, reasoning RL and RL for agents.',
         'Nine years, seven overlapping threads. Each one is still in use; new stages were added on top of the old ones, not in their place.')

# --- the RLHF pipeline ----------------------------------------------------------------------
b = [text(10, 22, 'Classic RLHF in three stages', 't-title'),
     box(20, 50, 210, 70, 'box-1', ['1. Supervised fine-tuning', 'copy answers people wrote']),
     box(255, 50, 210, 70, 'box-2', ['2. Reward model', 'learn which answer people prefer']),
     box(490, 50, 210, 70, 'box-3', ['3. RL with PPO', 'raise the reward, stay close']),
     arrow([(230, 85), (255, 85)]), arrow([(465, 85), (490, 85)]),
     box(20, 160, 210, 60, 'box', ['data: prompt + good answer', 'InstructGPT: about 13k prompts']),
     box(255, 160, 210, 60, 'box', ['data: prompt + ranked answers', 'InstructGPT: 33k prompts']),
     box(490, 160, 210, 60, 'box', ['data: prompts only', 'InstructGPT: 31k prompts']),
     f'<line class="edge-dim" x1="125" y1="120" x2="125" y2="160"/>',
     f'<line class="edge-dim" x1="360" y1="120" x2="360" y2="160"/>',
     f'<line class="edge-dim" x1="595" y1="120" x2="595" y2="160"/>',
     box(255, 260, 445, 54, 'box-on', ['score = r(x, y) − β · log( π(y | x) / π_SFT(y | x) )']),
     arrow([(360, 220), (360, 260)], 'edge-on', 'arrow-on'),
     arrow([(595, 260), (595, 220)], 'edge-on', 'arrow-on'),
     text(20, 280, 'The SFT model is also the', 't-muted'), text(20, 298, 'reference that the KL term', 't-muted'),
     text(20, 316, 'measures distance from.', 't-muted')]
svg('p4-rlhf', 720, 336, 'The three stages of RLHF: supervised fine-tuning, a reward model trained on ranked answers, and PPO training against the reward model with a KL penalty.', b,
    'The InstructGPT recipe of 2022, with its dataset sizes. Stage 3 needs no new human labels: the reward model scores every answer.')

# --- reward model -------------------------------------------------------------------------------
b = [text(10, 22, 'A reward model: a language model with a one-number head', 't-title'),
     box(20, 50, 200, 50, 'box-1', ['prompt x + answer y_w', '(the one people chose)']),
     box(20, 150, 200, 50, 'box-4', ['prompt x + answer y_l', '(the one they rejected)']),
     box(260, 50, 170, 50, 'box', ['Transformer', 'from the SFT model']),
     box(260, 150, 170, 50, 'box', ['same Transformer', 'same weights']),
     box(470, 50, 90, 50, 'box-on', ['r_w = 2.0', 'linear head']),
     box(470, 150, 90, 50, 'box-on', ['r_l = 0.0', 'linear head']),
     arrow([(220, 75), (260, 75)]), arrow([(220, 175), (260, 175)]),
     arrow([(430, 75), (470, 75)]), arrow([(430, 175), (470, 175)]),
     box(590, 90, 115, 70, 'box-3', ['σ(2.0 − 0.0)', '= 0.881', 'loss 0.127']),
     arrow([(560, 75), (590, 110)]), arrow([(560, 175), (590, 140)]),
     text(20, 240, 'The head reads the hidden state at the last token and outputs one number. Only the difference', 't-muted'),
     text(20, 258, 'between two scores matters, so the loss teaches an ordering, not an absolute grade.', 't-muted')]
svg('p4-rm', 720, 276, 'Two answers to the same prompt go through the same Transformer and a linear head, giving scores 2.0 and 0.0. '
    'The sigmoid of the difference is 0.881 and the loss is 0.127.', b,
    'The Bradley-Terry reward model. The numbers are the worked example from the text, computed by our script.')

# --- best-of-n chart ---------------------------------------------------------------------
b = [text(10, 22, 'Best-of-n: more samples, more chance of a right one, more distance from the model', 't-title')]
ns = [1, 2, 4, 8, 16, 64]
win = [0.3, 0.51, 0.7599, 0.9424, 0.9967, 1.0]
kl = [0.0, 0.1931, 0.6363, 1.2044, 1.8351, 3.1745]
X0, X1, Y0, Y1 = 70, 660, 50, 250
xs = lambda i: X0 + i / 5 * (X1 - X0)
for t in (0, 0.25, 0.5, 0.75, 1.0):
    y = Y1 - t * (Y1 - Y0)
    b.append(f'<line class="grid" x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}"/>')
    b.append(text(X0 - 8, y + 4, f'{t:g}', 't-tick', 'end'))
    b.append(text(X1 + 8, y + 4, f'{t * 4:g}', 't-tick'))
b.append(f'<line class="axis" x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}"/>')
for i, n in enumerate(ns):
    b.append(text(xs(i), Y1 + 18, f'n = {n}', 't-tick', 'middle'))
b.append(f'<polyline class="l1" points="{" ".join(f"{xs(i):.1f},{Y1 - w * (Y1 - Y0):.1f}" for i, w in enumerate(win))}"/>')
b.append(f'<polyline class="l2" points="{" ".join(f"{xs(i):.1f},{Y1 - k / 4 * (Y1 - Y0):.1f}" for i, k in enumerate(kl))}"/>')
for i in range(6):
    b.append(f'<circle class="s1" cx="{xs(i):.1f}" cy="{Y1 - win[i] * (Y1 - Y0):.1f}" r="4"/>')
    b.append(f'<circle class="s2" cx="{xs(i):.1f}" cy="{Y1 - kl[i] / 4 * (Y1 - Y0):.1f}" r="4"/>')
b += [text(X0, Y1 + 42, '● chance that at least one of n answers is right, if each is right with p = 0.3 (left axis)', 't-s1'),
      text(X0, Y1 + 62, '● KL of best-of-n from the model, log n − (n − 1)/n, in nats (right axis)', 't-s2')]
svg('p4-bon', 720, 330, 'A line chart over n equal to 1, 2, 4, 8, 16 and 64. The chance of at least one right answer rises from 0.3 to almost 1. '
    'The KL distance rises from 0 to 3.17 nats.', b,
    'Picking the best of n answers is the simplest way to optimise a reward. Its cost in KL grows like log n, so it is a good yardstick for how hard RL pushes a model. Numbers from our script.')

# --- PPO vs DPO -------------------------------------------------------------------------------
b = [text(10, 22, 'What each method keeps in memory during training', 't-title'),
     text(180, 50, 'RLHF with PPO: four models', 't-note', 'middle'),
     text(540, 50, 'DPO: two models', 't-note', 'middle'),
     box(30, 66, 140, 54, 'box-on', ['policy', 'trained']),
     box(190, 66, 140, 54, 'box', ['reference', 'frozen']),
     box(30, 136, 140, 54, 'box-3', ['reward model', 'frozen']),
     box(190, 136, 140, 54, 'box-on', ['value model', 'trained']),
     box(400, 66, 140, 54, 'box-on', ['policy', 'trained']),
     box(560, 66, 140, 54, 'box', ['reference', 'frozen']),
     f'<line class="edge-dim" x1="360" y1="40" x2="360" y2="290"/>',
     text(30, 220, 'Loop: sample answers, score them,', 't-muted'),
     text(30, 238, 'estimate advantages, run PPO steps,', 't-muted'),
     text(30, 256, 'sample again. Online and fiddly.', 't-muted'),
     text(400, 150, 'Loop: read a fixed pair (chosen, rejected),', 't-muted'),
     text(400, 168, 'compute four log-probabilities,', 't-muted'),
     text(400, 186, 'one classification-style loss. Offline.', 't-muted'),
     box(400, 210, 300, 60, 'box-2', ['loss = −log σ( β·[Δ chosen] − β·[Δ rejected] )', 'Δ = log π − log π_ref of the answer'])]
svg('p4-dpo', 720, 296, 'Left: PPO-based RLHF holds a policy, a reference, a reward model and a value model. Right: DPO holds only a policy and a reference.', b,
    'DPO drops the reward model and the value model and needs no sampling during training. The price: it learns only from the pairs in its data set.')

# --- ORM vs PRM -------------------------------------------------------------------------------
b = [text(10, 22, 'Outcome reward vs process reward for one solution', 't-title')]
steps = ['Step 1: 17 × 24 = 17 × 20 + 17 × 4', 'Step 2: 17 × 20 = 340', 'Step 3: 17 × 4 = 58', 'Step 4: 340 + 58 = 398']
for i, s in enumerate(steps):
    y = 48 + i * 46
    b.append(box(20, y, 330, 38, 'box-4' if i >= 2 else 'box', []))
    b.append(text(34, y + 24, s, 't-mono'))
    b.append(text(470, y + 24, ['✓ correct', '✓ correct', '✗ wrong', '✗ follows the error'][i], 't-on' if i < 2 else 't-bad', 'middle'))
b += [text(470, 40, 'process reward model', 't-note', 'middle'),
      text(640, 40, 'outcome reward', 't-note', 'middle'),
      box(590, 48, 100, 176, 'box-4', ['answer 398', 'is wrong:', 'reward 0', 'for all']),
      text(20, 254, 'The outcome label says only "wrong". The process labels say where: step 3. That is a far more useful signal.', 't-muted')]
svg('p4-prm', 720, 272, 'A four-step solution of 17 times 24. A process reward model marks steps 1 and 2 correct and step 3 wrong. An outcome reward model only sees that the final answer 398 is wrong.', b,
    'The correct answer is 408. Lightman and colleagues paid people to label about 800,000 steps like these (PRM800K).')

# --- GRPO group -----------------------------------------------------------------------------
b = [text(10, 22, 'GRPO: sample a group, score it, compare each answer with the group', 't-title'),
     box(20, 120, 130, 70, 'box-2', ['one maths', 'question']),
     box(190, 120, 120, 70, 'box-on', ['policy', 'samples 8']),
     arrow([(150, 155), (190, 155)])]
R = [1, 0, 0, 1, 1, 0, 0, 0]
A = [1.291, -0.775, -0.775, 1.291, 1.291, -0.775, -0.775, -0.775]
for i, (r, a) in enumerate(zip(R, A)):
    y = 40 + i * 28
    b.append(f'<rect class="{"box-1" if r else "box-4"}" x="350" y="{y}" width="110" height="22" rx="4"/>')
    b.append(text(405, y + 16, f'answer {i + 1}: {"right" if r else "wrong"}', 't-muted', 'middle'))
    b.append(text(490, y + 16, f'r = {r}', 't-note', 'middle'))
    b.append(text(600, y + 16, f'A = {a:+.3f}'.replace('-', '−'), 't-s1' if a > 0 else 't-bad', 'middle'))
    b.append(f'<line class="edge-dim" x1="310" y1="155" x2="350" y2="{y + 11}"/>')
b += [text(350, 284, 'mean = 0.375, std = 0.484, A = (r − mean) / std', 't-note'),
      text(20, 312, 'No value network: the group mean is the baseline. Right answers are pushed up, wrong ones down, then a PPO-style clipped step.', 't-muted')]
svg('p4-grpo', 720, 330, 'A question goes to the policy, which samples eight answers. Three are right with reward 1 and five are wrong with reward 0. '
    'Advantages are plus 1.291 for the right ones and minus 0.775 for the wrong ones.', b,
    'Our script\'s group of eight. If all eight answers were right (or all wrong), every advantage would be 0, and the question would teach nothing.')

# --- the 2026 pipeline ----------------------------------------------------------------------
b = [text(10, 22, 'How a frontier language model is trained, as far as public reports show', 't-title')]
stages = [('Pre-training', 'trillions of tokens', 'next-token loss', 'box-1'),
          ('Mid-training', 'curated maths, code,', 'reasoning, long context', 'box-1'),
          ('SFT', 'instructions, chats,', 'reasoning traces, tools', 'box-2'),
          ('Preference tuning', 'reward models, DPO,', 'AI feedback', 'box-2'),
          ('Reasoning RL', 'verifiable rewards:', 'tests, checkers', 'box-3'),
          ('Agent RL', 'multi-step tasks', 'in tool environments', 'box-3'),
          ('Safety training', 'specs, constitutions,', 'red-team data', 'box-4'),
          ('Distillation', 'small models learn', 'from the big one', 'box-4')]
for i, (n, a, c, cls) in enumerate(stages):
    col, row = i % 4, i // 4
    x, y = 20 + col * 175, 50 + row * 130
    b.append(box(x, y, 155, 86, cls, [n, a, c]))
    if col < 3:
        b.append(arrow([(x + 155, y + 43), (x + 175, y + 43)]))
b.append(arrow([(637, 136), (637, 158), (97, 158), (97, 180)]))
b += [f'<rect class="box-ghost" x="20" y="300" width="680" height="44" rx="6"/>',
      text(360, 327, 'Evaluation at every stage: benchmarks, held-out tests, human and model judges, safety tests', 't-note', 'middle'),
      text(20, 372, 'Stages 3 to 7 are often repeated in rounds, and their order differs between labs. Few labs publish their full recipe.', 't-muted')]
svg('p4-pipeline', 720, 390, 'Eight stages in two rows: pre-training, mid-training, supervised fine-tuning, preference tuning, reasoning RL, agent RL, safety training and distillation, with evaluation running under all of them.', b,
    'A composite drawn from the reports of Llama 3, DeepSeek-R1 and V3.2, Qwen3, Olmo 3, Kimi K2 and Gemini 2.5. No single lab describes exactly this list.')

# =============================================================================================
# Part 5: inside a modern language model (numbers from pretrain_history.py, results/pretrain_history.json)
# =============================================================================================
P5 = json.load(open(os.path.join(HERE, 'results', 'pretrain_history.json')))

timeline('p5-timeline', 'How the pre-trained model took its 2026 shape, 2017 to 2026', 2016.5, 2026.9,
         [2017, 2019, 2021, 2023, 2025],
         [('Transformer, GPT, BERT', '2017 to 2020', 2017.45, 2020.4, 'box-1'),
          ('Scaling laws: Kaplan, Chinchilla', '2020 to 2022', 2020.05, 2022.25, 'box-2'),
          ('Block: RMSNorm, SwiGLU, RoPE', '2019 to 2021', 2019.8, 2021.3, 'box-3'),
          ('Mixture of experts at scale', '2017 to 2026', 2017.05, 2026.8, 'box-4'),
          ('Open web data: C4 to FineWeb', '2019 to 2024', 2019.8, 2024.5, 'box-1'),
          ('KV savings: MQA, GQA, MLA', '2019 to 2024', 2019.85, 2024.4, 'box-2'),
          ('Long context, FP8, multi-token', '2023 to 2026', 2023.5, 2026.8, 'box-3')],
         'A timeline from 2017 to 2026 with seven threads: the Transformer, scaling laws, changes to the block, mixture of experts, '
         'open web data sets, ways to shrink the KV cache, and long context with FP8 training.',
         'The core of the 2017 design is still there. Most changes after 2020 make the same model cheaper to train, cheaper to serve, or able to read longer inputs.')

# --- the data pipeline ------------------------------------------------------------------------
b = [text(10, 22, 'From raw web pages to training tokens', 't-title')]
steps = [('Common Crawl', 'raw HTML, monthly'), ('Extract text', 'drop menus, ads, code'),
         ('Language ID', 'keep the target languages'), ('Heuristic filters', 'length, repeats, spam'),
         ('Deduplicate', 'URL, MinHash, lines'), ('Quality classifier', 'keep the best pages'),
         ('Mix and weight', 'web, code, maths'), ('Tokenise', 'text becomes integer IDs')]
for i, (n, s) in enumerate(steps):
    col, row = i % 4, i // 4
    x, y = 20 + col * 175, 46 + row * 120
    b.append(box(x, y, 155, 62, 'box-on' if i in (4, 5) else ('box-2' if i == 0 else ('box-1' if i == 7 else 'box')), [n, s]))
    if col < 3:
        b.append(arrow([(x + 155, y + 31), (x + 175, y + 31)]))
b.append(arrow([(637, 108), (637, 136), (97, 136), (97, 166)]))
b += [text(20, 270, 'FineWeb (2024): 96 Common Crawl snapshots in, 15 trillion tokens out. FineWeb-Edu keeps 1.3 trillion of them.', 't-muted'),
      text(20, 290, 'Llama 3 mix (2024): about 50% general knowledge, 25% maths and reasoning, 17% code, 8% other languages.', 't-muted')]
svg('p5-data', 720, 306, 'Eight steps in two rows: Common Crawl, text extraction, language identification, heuristic filters, '
    'deduplication, a quality classifier, mixing, and tokenisation.', b,
    'Each step throws data away. Deduplication and the quality classifier (highlighted) changed results the most in the open studies of 2023 and 2024.')

# --- BPE merges -------------------------------------------------------------------------------
b = [text(10, 22, 'Byte-pair encoding: merge the most frequent pair, again and again', 't-title'),
     text(20, 50, 'words and counts:  low ×5,  lower ×2,  newest ×6,  widest ×3       (</w> marks the end of a word)', 't-muted')]
rows = [('start', 'l o w </w>', 'l o w e r </w>', 'n e w e s t </w>', 'w i d e s t </w>'),
        ('1: e + s (9)', 'l o w </w>', 'l o w e r </w>', 'n e w es t </w>', 'w i d es t </w>'),
        ('2: es + t (9)', 'l o w </w>', 'l o w e r </w>', 'n e w est </w>', 'w i d est </w>'),
        ('3: est + </w> (9)', 'l o w </w>', 'l o w e r </w>', 'n e w est</w>', 'w i d est</w>'),
        ('4: l + o (7)', 'lo w </w>', 'lo w e r </w>', 'n e w est</w>', 'w i d est</w>'),
        ('5: lo + w (7)', 'low </w>', 'low e r </w>', 'n e w est</w>', 'w i d est</w>'),
        ('8: new + est</w> (6)', 'low </w>', 'low e r </w>', 'newest</w>', 'w i d est</w>')]
xs = [20, 172, 292, 432, 574]
b += [text(xs[0], 78, 'merge (count)', 't-label')] + [text(x, 78, h, 't-label') for x, h in zip(xs[1:], ['low', 'lower', 'newest', 'widest'])]
for r, row in enumerate(rows):
    y = 104 + r * 30
    if r == len(rows) - 1:
        b.append(text(20, y - 4, '... merges 6 and 7: n + e, ne + w ...', 't-muted'))
        y += 26
    b.append(f'<rect class="{"box-on" if r in (3, 6) else "box"}" x="14" y="{y - 18}" width="692" height="26" rx="4"/>')
    for x, s in zip(xs, row):
        b.append(text(x, y, s, 't-note' if x == 20 else 't-mono'))
b.append(text(20, 352, 'After 10 merges, "lowest" (never seen) becomes  low + est</w>.  The vocabulary grows by one symbol per merge.', 't-muted'))
svg('p5-bpe', 720, 368, 'A table of byte-pair merges on four words. e and s merge first, then es and t, then est and the end marker, '
    'then l and o, then lo and w. Later, new and est end-of-word join into newest.', b,
    'The toy vocabulary of Sennrich, Haddow and Birch (2016), run by our script with the paper\'s own algorithm. Real vocabularies, built on bytes instead of letters, hold 32,000 to 262,000 tokens.')

# --- Chinchilla iso-FLOP chart -------------------------------------------------------------------
b = [text(10, 22, 'Iso-FLOP curves from the Chinchilla fit: each budget has a best model size', 't-title')]
X0, X1, Y0, Y1 = 70, 690, 44, 300
lx = lambda N: X0 + (math.log10(N) - 7.5) / (12 - 7.5) * (X1 - X0)
ly = lambda Lv: Y1 - (Lv - 1.85) / (2.75 - 1.85) * (Y1 - Y0)
for t in [1.9, 2.1, 2.3, 2.5, 2.7]:
    b.append(f'<line class="grid" x1="{X0}" y1="{ly(t):.1f}" x2="{X1}" y2="{ly(t):.1f}"/>')
    b.append(text(X0 - 8, ly(t) + 4, f'{t:.1f}', 't-tick', 'end'))
for e, lab in [(8, '100M'), (9, '1B'), (10, '10B'), (11, '100B'), (12, '1T')]:
    b.append(f'<line class="grid" x1="{lx(10 ** e):.1f}" y1="{Y0}" x2="{lx(10 ** e):.1f}" y2="{Y1}"/>')
    b.append(text(lx(10 ** e), Y1 + 18, lab, 't-tick', 'middle'))
b += [f'<line class="axis" x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}"/>', f'<line class="axis" x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}"/>',
      text(X1, Y1 + 38, 'parameters N (log scale); tokens D = C / (6N)', 't-muted', 'end'), text(14, Y0 - 6, 'loss', 't-muted')]
cls = {'1e+21': ('l3', 's3', 't-s3'), '1e+22': ('l1', 's1', 't-s1'), '1e+23': ('l4', 's4', 't-s4'), '5.76e+23': ('key-line', 's2', 't-s2')}
names = {'1e+21': '10²¹ FLOPs', '1e+22': '10²²', '1e+23': '10²³', '5.76e+23': '5.76 × 10²³ (Gopher, Chinchilla)'}
for k, c in P5['iso'].items():
    pts = [(lx(N), ly(Lv)) for N, Lv in zip(c['N'], c['L']) if 1.85 <= Lv <= 2.75]
    b.append(f'<polyline class="{cls[k][0]}" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>')
    bx, by = lx(c['best_N']), ly(c['best_L'])
    b.append(f'<circle class="{cls[k][1]}" cx="{bx:.1f}" cy="{by:.1f}" r="5"/>')
    b.append(text(bx + 8, by + 18, names[k], cls[k][2]))
b.append(text(X0 + 10, Y1 - 12, 'dot = lowest loss for that budget', 't-muted'))
svg('p5-iso', 720, 352, 'Four U-shaped curves of loss against model size, one per compute budget from 10 to the 21 to 5.76 times 10 to the 23 FLOPs. '
    'The minimum moves to larger models as the budget grows: 1.84 billion, 5.2 billion, 14.7 billion and 32.2 billion parameters.', b,
    'Drawn from the paper\'s fitted formula L(N, D) = 1.69 + 406.4/N^0.34 + 410.7/D^0.28, not from its measured runs. Too small a model starves on parameters; too large a model starves on tokens.')

# --- the 2017 block vs the 2026 block ----------------------------------------------------------
def block(ox, title, rows, notes):
    out = [text(ox + 150, 44, title, 't-note', 'middle')]
    y = 96 + sum(h + 16 for _, _, h in rows)          # bottom of the stack; the data flows upward
    out.append(text(ox + 150, y + 22, 'tokens in', 't-muted', 'middle'))
    for lab, cls_, h in rows:
        y -= h + 16
        out.append(box(ox + 50, y, 200, h, cls_, lab))
        out.append(arrow([(ox + 150, y + h + 16), (ox + 150, y + h)]))
    out.append(arrow([(ox + 150, y), (ox + 150, y - 16)]))
    out.append(text(ox + 150, y - 22, 'next layer', 't-muted', 'middle'))
    for i, s in enumerate(notes):
        out.append(text(ox + 10, 462 + i * 18, s, 't-muted'))
    return out


b = [text(10, 22, 'One Transformer layer, 2017 and 2026', 't-title')]
b += block(20, '2017: "Attention Is All You Need"',
           [(['+ sine positions', '(added once, at the input)'], 'box', 40), (['Multi-head attention', 'every head has its own K, V'], 'box-2', 40),
            (['Add, then LayerNorm', '(post-norm)'], 'box', 40), (['Feed-forward', 'ReLU, 4d wide'], 'box-3', 40),
            (['Add, then LayerNorm'], 'box', 30)],
           ['Dense: every token uses every weight.', 'Encoder and decoder, for translation.', 'Base model: 65M parameters.'])
b += block(380, '2026: a typical decoder (Llama 3, DeepSeek-V3)',
           [(['RMSNorm', '(pre-norm, before the block)'], 'box-on', 40), (['Attention with RoPE', 'GQA or MLA: shared K, V'], 'box-2', 40),
            (['Add (residual)'], 'box', 30), (['RMSNorm'], 'box-on', 30), (['SwiGLU feed-forward,', 'often a mixture of experts'], 'box-3', 40),
            (['Add (residual)'], 'box', 30)],
           ['Decoder only; 128K to 1M+ tokens.', 'Often sparse: a few experts per token.', 'Llama 3 405B: 126 layers deep.'])
b.append(f'<line class="edge-dim" x1="360" y1="36" x2="360" y2="510"/>')
svg('p5-blocks', 720, 528, 'Two stacks side by side. The 2017 layer adds sine positions, runs multi-head attention, then add and LayerNorm, '
    'a ReLU feed-forward, then add and LayerNorm. The 2026 layer runs RMSNorm, attention with RoPE and shared keys and values, '
    'a residual add, RMSNorm, a SwiGLU or mixture-of-experts feed-forward, and another residual add.', b,
    'Same skeleton, new parts. The norm moved before each block, the positions moved into attention, the keys and values are shared, and the feed-forward got a gate and often a router.')

# --- RoPE rotation ----------------------------------------------------------------------------
b = [text(10, 22, 'RoPE: turn each pair of numbers by an angle that grows with position', 't-title')]
def plane(cx, cy, r, vecs, title):
    out = [text(cx, 50, title, 't-note', 'middle'),
           f'<line class="axis" x1="{cx - r - 10}" y1="{cy}" x2="{cx + r + 10}" y2="{cy}"/>',
           f'<line class="axis" x1="{cx}" y1="{cy + r + 10}" x2="{cx}" y2="{cy - r - 10}"/>',
           f'<circle class="box-ghost" cx="{cx}" cy="{cy}" r="{r}"/>']
    for ang, lab, c, h in vecs:
        x, y = cx + r * math.cos(ang), cy - r * math.sin(ang)
        out.append(f'<line class="{c}" x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}"/>')
        out.append(head(x, y, x - cx, y - cy, h))
        out.append(text(x + 10 * math.cos(ang), y - 10 * math.sin(ang) + 4, lab, 't-note', 'start' if math.cos(ang) >= 0 else 'end'))
    return out


th = 0.3
aq, ak = math.atan2(0.5, 1.0), math.atan2(-0.2, 0.8)
b += plane(190, 190, 105, [(aq + 3 * th, 'q at position 3', 'edge-on', 'arrow-on'), (ak + 1 * th, 'k at position 1', 'edge-2', 'arrow-2')],
           'positions 3 and 1')
b += plane(530, 190, 105, [(aq + 7 * th, 'q at 7', 'edge-on', 'arrow-on'), (ak + 5 * th, 'k at 5', 'edge-2', 'arrow-2')],
           'positions 7 and 5: both turned 4θ more')
b += [text(360, 340, 'The angle between q and k is the same in both pictures, so the dot product is the same: 0.2389.', 't-note', 'middle'),
      text(360, 362, 'Attention sees only the distance, 2 tokens, not where the pair sits. (θ = 0.3, from our script.)', 't-muted', 'middle')]
svg('p5-rope', 720, 380, 'Two circles. On the left, a query rotated for position 3 and a key rotated for position 1. On the right, the same vectors '
    'rotated for positions 7 and 5. The angle between them is the same in both, so the dot product is 0.2389 in both.', b,
    'A real head has 64 such pairs, each turning at its own speed, from one radian per token down to a full turn every 54,000 tokens (base 10,000) or 2.6 million tokens (base 500,000).')

# --- KV cache bar chart ------------------------------------------------------------------------
b = [text(10, 22, 'KV cache for one token, all layers, 16-bit numbers', 't-title')]
kvrows = [('Llama 3 70B shape, MHA (64 KV heads)', 2560.0, 'box-4'), ('Llama 3 70B, GQA (8 KV heads, as built)', 320.0, 'box-1'),
          ('Llama 3 70B shape, MQA (1 KV head)', 40.0, 'box'), ('DeepSeek-V3 shape, MHA (128 heads)', 3904.0, 'box-4'),
          ('DeepSeek-V3, MLA (576 numbers a layer, as built)', 68.6, 'box-1')]
X0, X1 = 330, 560
for i, (n, v, c) in enumerate(kvrows):
    y = 44 + i * 44 + (14 if i >= 3 else 0)
    b.append(text(10, y + 20, n, 't-note'))
    w = max(3, v / 3904 * (X1 - X0))
    b.append(f'<rect class="{c}" x="{X0}" y="{y}" width="{w:.1f}" height="28" rx="4"/>')
    gib = {2560.0: '320 GiB', 320.0: '40 GiB', 40.0: '5 GiB', 3904.0: '488 GiB', 68.6: '8.6 GiB'}[v]
    b.append(text(X0 + w + 8, y + 19, f'{v:,.1f} KiB'.replace('.0 KiB', ' KiB'), 't-val'))
    b.append(text(712, y + 19, gib, 't-muted', 'end'))
b += [text(712, 40, 'at 128K tokens', 't-muted', 'end'),
      text(10, 300, 'GQA cuts the Llama cache 8 times. MLA cuts the DeepSeek cache 56.9 times, because it stores one small latent', 't-muted'),
      text(10, 318, 'vector per layer and rebuilds the keys and values from it. An 80 GB GPU cannot even hold the MHA cache for one long chat.', 't-muted')]
svg('p5-kv', 720, 334, 'A bar chart of KV-cache size per token. Llama 3 70B shape: 2,560 KiB with multi-head attention, 320 KiB with grouped-query attention, '
    '40 KiB with multi-query attention. DeepSeek-V3 shape: 3,904 KiB with multi-head attention and 68.6 KiB with multi-head latent attention.', b,
    'Bytes per token = 2 (keys and values) × layers × KV heads × head size × 2 bytes. The MHA rows are what the same shapes would need without sharing. Numbers from our script.')

# --- MoE router ---------------------------------------------------------------------------------
b = [text(10, 22, 'A mixture-of-experts layer: a router picks 2 of 8 experts for each token', 't-title'),
     box(20, 140, 110, 56, 'box-2', ['token x', 'hidden vector']),
     box(160, 140, 110, 56, 'box-on', ['router', 'softmax(x W_r)']),
     arrow([(130, 168), (160, 168)])]
probs = [0.4528, 0.1666, 0.101, 0.0677, 0.0454, 0.0225, 0.0613, 0.0827]
for i, p_ in enumerate(probs):
    y = 44 + i * 36
    on = i < 2
    b.append(f'<rect class="{"box-on" if on else "box"}" x="330" y="{y}" width="{max(4, p_ * 250):.1f}" height="24" rx="3"/>')
    b.append(text(322, y + 17, f'E{i}', 't-note', 'end'))
    b.append(text(336 + max(4, p_ * 250), y + 17, f'{p_:.4f}', 't-val' if on else 't-muted'))
    b.append(f'<line class="{"edge-on" if on else "edge-dim"}" x1="270" y1="168" x2="300" y2="{y + 12}"/>')
b += [box(560, 50, 140, 50, 'box-3', ['FFN expert 0', 'gate 0.7311']),
      box(560, 120, 140, 50, 'box-3', ['FFN expert 1', 'gate 0.2689']),
      arrow([(506, 56), (560, 70)], 'edge-on', 'arrow-on'), arrow([(434, 92), (560, 140)], 'edge-on', 'arrow-on'),
      box(522, 220, 194, 56, 'box-1', ['output', '0.7311 E0(x) + 0.2689 E1(x)']),
      arrow([(700, 75), (710, 75), (710, 220)]), arrow([(630, 170), (630, 220)]),
      text(20, 340, 'The six other experts do no work for this token. The two chosen scores, 0.4528 and 0.1666, are rescaled to add up to 1.', 't-muted'),
      text(20, 358, 'Each token picks its own pair, so all experts are busy over a batch, but each token pays for only two.', 't-muted')]
svg('p5-moe', 720, 374, 'A token goes to a router, which gives eight expert probabilities. Experts 0 and 1 have the highest, 0.4528 and 0.1666. '
    'They get gate weights 0.7311 and 0.2689, and the output is their weighted sum.', b,
    'One token of our script\'s router. Mixtral uses exactly this shape: 8 experts, top 2. DeepSeek-V3 picks 8 of 256 routed experts, plus 1 shared expert that every token uses.')

# --- parallelism -------------------------------------------------------------------------------
b = [text(10, 22, 'Three ways to split one training run across many GPUs', 't-title')]
# data parallel
b += [text(120, 50, 'Data parallel', 't-note', 'middle'), text(120, 66, 'same model, different data', 't-muted', 'middle')]
for i in range(3):
    y = 82 + i * 62
    b.append(box(20, y, 200, 46, 'box-1', [f'GPU {i}: full model', f'batch slice {i}']))
b += [box(14, 274, 212, 40, 'box-on', ['all-reduce: average gradients'])]
# tensor parallel
b += [text(360, 50, 'Tensor parallel', 't-note', 'middle'), text(360, 66, 'split each weight matrix', 't-muted', 'middle')]
for i in range(4):
    b.append(f'<rect class="box-2" x="{260 + i * 50}" y="90" width="46" height="120" rx="4"/>')
    b.append(text(283 + i * 50, 154, f'GPU {i}', 't-muted', 'middle'))
b += [text(360, 230, 'one 4096 × 14336 matrix', 't-muted', 'middle'), text(360, 248, '→ four 4096 × 3584 slices', 't-muted', 'middle'),
      box(270, 274, 180, 40, 'box-on', ['exchange after each layer'])]
# pipeline parallel
b += [text(600, 50, 'Pipeline parallel', 't-note', 'middle'), text(600, 66, 'split the layers into stages', 't-muted', 'middle')]
for i in range(4):
    y = 82 + i * 46
    b.append(text(500, y + 22, f'GPU {i}', 't-muted'))
    for m in range(4):
        x = 538 + (i + m) * 24
        b.append(f'<rect class="box-3" x="{x}" y="{y + 6}" width="21" height="24" rx="3"/>')
        b.append(text(x + 10.5, y + 23, str(m + 1), 't-tick', 'middle'))
b += [text(600, 280, 'blank slots = idle time,', 't-muted', 'middle'), text(600, 298, 'the "bubble": 3/7 here', 't-muted', 'middle'),
      f'<line class="edge-dim" x1="240" y1="40" x2="240" y2="320"/>', f'<line class="edge-dim" x1="480" y1="40" x2="480" y2="320"/>',
      text(10, 346, 'Large runs use all three at once, plus expert parallelism for MoE layers and ZeRO / FSDP to shard optimizer states.', 't-muted')]
svg('p5-parallel', 720, 362, 'Three panels. Data parallel: three GPUs each hold the full model and see a different slice of the batch, then average gradients. '
    'Tensor parallel: one weight matrix is cut into four column slices on four GPUs. Pipeline parallel: four GPUs hold four stages of layers, '
    'and four micro-batches flow through them in a staircase.', b,
    'With 4 stages and 4 micro-batches the pipeline is idle 3/7 of the time; with 32 micro-batches only 3/35 (8.6%). Numbers from our script.')

# --- LR schedule -------------------------------------------------------------------------------
b = [text(10, 22, 'Learning rate: linear warm-up, then cosine decay (Llama 3 405B settings)', 't-title')]
X0, X1, Y0, Y1 = 80, 690, 44, 250
pts = sorted(P5['lr'])
sx = lambda t: X0 + t / 1.2e6 * (X1 - X0)
sy = lambda v: Y1 - v / 8.4e-5 * (Y1 - Y0)
for v in [0, 2e-5, 4e-5, 6e-5, 8e-5]:
    b.append(f'<line class="grid" x1="{X0}" y1="{sy(v):.1f}" x2="{X1}" y2="{sy(v):.1f}"/>')
    b.append(text(X0 - 8, sy(v) + 4, '0' if v == 0 else f'{v * 1e5:g}e-5', 't-tick', 'end'))
for t in [0, 3e5, 6e5, 9e5, 1.2e6]:
    b.append(text(sx(t), Y1 + 18, f'{t / 1e3:,.0f}K' if t else '0', 't-tick', 'middle'))
b.append(f'<polyline class="l1" points="{" ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in pts)}"/>')
for t, v, lab in [(8000, 8e-5, 'peak 8e-5 after 8,000 warm-up steps'), (604000, 4.04e-5, 'half-way: 4.04e-5'), (1.2e6, 8e-7, 'end: 8e-7')]:
    b.append(f'<circle class="s1" cx="{sx(t):.1f}" cy="{sy(v):.1f}" r="4"/>')
    b.append(text(sx(t) + (10 if t < 1e6 else -10), sy(v) - 10, lab, 't-s1', 'start' if t < 1e6 else 'end'))
b += [f'<line class="axis" x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}"/>', text(X1, Y1 + 38, 'training step', 't-muted', 'end')]
svg('p5-lr', 720, 300, 'A curve that rises in a straight line to 8 times 10 to the minus 5 at step 8,000, then falls along a cosine to 8 times 10 to the minus 7 at step 1.2 million.', b,
    'The warm-up is too short to see at this scale: 8,000 of 1,200,000 steps. Llama 3 then annealed the rate to 0 over the last 40 million tokens. Curve from our script.')

# --- multimodal token stream ---------------------------------------------------------------------
b = [text(10, 22, 'One sequence, many kinds of token', 't-title'),
     box(20, 50, 170, 70, 'box-2', ['image 224 × 224', 'cut into 16 × 16 patches']),
     box(20, 140, 170, 56, 'box-3', ['audio', 'short frames → vectors']),
     box(20, 216, 170, 56, 'box-1', ['text', 'BPE tokens → vectors']),
     box(230, 50, 150, 70, 'box', ['196 patch tokens', 'linear projection']),
     arrow([(190, 85), (230, 85)]),
     box(230, 140, 150, 56, 'box', ['audio tokens', 'an audio encoder']), arrow([(190, 168), (230, 168)]),
     box(230, 216, 150, 56, 'box', ['text tokens', 'embedding table']), arrow([(190, 244), (230, 244)])]
seq = ['t', 't', 'i', 'i', 'i', 'i', 'i', 't', 't', 'a', 'a', 'a', 't', 't']
for j, s in enumerate(seq):
    c = {'t': 'box-1', 'i': 'box-2', 'a': 'box-3'}[s]
    b.append(f'<rect class="{c}" x="{420 + j * 20}" y="150" width="17" height="36" rx="3"/>')
b += [arrow([(380, 85), (440, 146)]), arrow([(380, 168), (420, 168)]), arrow([(380, 244), (440, 190)]),
      box(420, 230, 280, 50, 'box-on', ['one Transformer reads it all', 'same attention, same layers']),
      arrow([(560, 186), (560, 230)]),
      text(470, 136, 'the token stream', 't-note'),
      text(20, 312, 'Gemini and Llama 4 are described as natively multimodal: they meet images (and, for Gemini, sound) in pre-training,', 't-muted'),
      text(20, 330, 'not only through an adapter bolted on afterwards. The exact encoders of the closed models are not public.', 't-muted')]
svg('p5-mm', 720, 346, 'Images are cut into patches, audio into frames and text into BPE tokens. Each becomes a vector, and the vectors are interleaved into one sequence that a single Transformer reads.', b,
    'A 224 × 224 image with 16 × 16 patches gives 196 tokens, the setting of the 2020 Vision Transformer. Modern models use bigger images and compress the patch tokens.')

# --- the full build ------------------------------------------------------------------------------
b = [text(10, 22, 'How a frontier model is built, end to end', 't-title')]
st = [('Data', 'crawl, filter, dedup,', 'mix, tokenise', 'box-2'), ('Pre-training', 'trillions of tokens,', 'next-token loss', 'box-1'),
      ('Mid-training', 'annealing on', 'high-quality data', 'box-1'), ('Long context', 'RoPE scaling,', '8K → 128K → 1M', 'box-1'),
      ('Post-training', 'SFT, preferences,', 'reasoning RL (Part 4)', 'box-3'), ('Distillation', 'small models learn', 'from the big one', 'box-3'),
      ('Serving', 'quantise, KV cache,', 'speculative decoding', 'box-4'), ('Model card', 'what the lab', 'chooses to publish', 'box')]
for i, (n, a1, a2, c) in enumerate(st):
    col, row = i % 4, i // 4
    x, y = 20 + col * 175, 50 + row * 130
    b.append(box(x, y, 155, 86, c, [n, a1, a2]))
    if col < 3:
        b.append(arrow([(x + 155, y + 43), (x + 175, y + 43)]))
b.append(arrow([(637, 136), (637, 158), (97, 158), (97, 180)]))
b += [f'<rect class="box-ghost" x="20" y="300" width="680" height="44" rx="6"/>',
      text(360, 327, 'Under it all: thousands of GPUs or TPUs, parallelism, bf16 or FP8, checkpoints and restarts', 't-note', 'middle'),
      text(20, 372, 'Parts 1 and 5 cover the first four boxes, Part 4 the post-training box. Labs differ in order and detail.', 't-muted')]
svg('p5-build', 720, 390, 'Eight stages in two rows: data, pre-training, mid-training, long context, post-training, distillation, serving, and the model card, with hardware and parallelism under all of them.', b,
    'A composite of the Llama 3, DeepSeek-V3, Gemini 2.5 and Olmo 3 reports. Closed labs confirm only parts of it.')


if __name__ == '__main__':
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(F, open(os.path.join(HERE, 'results', 'figs_rl.json'), 'w'), indent=0, ensure_ascii=False)
    print(len(F), 'figures:', ', '.join(F))
