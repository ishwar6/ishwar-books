"""Part 4: two key pieces of the released TerminalBench-2 harness (agent.py of meta-harness-tbench2-artifact),
run on our own machine without any model or API key.

  1. The environment bootstrap that Meta-Harness discovered: the same compound shell command, run locally,
     and its output parsed two ways (Method 1: the line-by-line parser of agent.py, rewritten; Method 2: one
     regular-expression split). We run it twice: in a folder named like the task folder /app, and with no /app.
  2. The completion check inherited from Terminus-KIRA: the agent must call task_complete twice in a row.
     We replay the state machine of agent.py on one action sequence, and the variant that the proposer tried
     in iteration 4 (do not reset the flag), to see the "verification spiral" of Appendix A.2.
Every printed line is saved to results/part4_tb2_stdout.txt."""
import os, re, subprocess, tempfile
from common import Log, save

# The command of AgentHarness._gather_env_snapshot, copied from agent.py. Only the folder name is a parameter.
BOOTSTRAP = (
    "echo '@@PWD@@' && pwd && "
    "echo '@@LS@@' && ls -la {app}/ 2>/dev/null && "
    "echo '@@LANG@@' && "
    "(python3 --version 2>&1 || echo 'python3: not found') && "
    "(gcc --version 2>&1 | head -1 || echo 'gcc: not found') && "
    "(g++ --version 2>&1 | head -1 || echo 'g++: not found') && "
    "(node --version 2>&1 || echo 'node: not found') && "
    "(java -version 2>&1 | head -1 || echo 'java: not found') && "
    "(rustc --version 2>&1 || echo 'rustc: not found') && "
    "(go version 2>&1 || echo 'go: not found') && "
    "echo '@@PKG@@' && "
    "(pip3 --version 2>&1 || echo 'pip3: not found') && "
    "(pip --version 2>&1 || echo 'pip: not found') && "
    "(apt-get --version 2>&1 | head -1 || echo 'apt-get: not found') && "
    "echo '@@MEM@@' && free -h 2>/dev/null | head -2 || true"
)


def parse_sections(stdout):
    """Method 1: the parser of agent.py. A line @@NAME@@ starts a new section; other lines go to the open one."""
    sections, key, lines = {}, None, []
    for line in stdout.split('\n'):
        if line.startswith('@@') and line.endswith('@@'):
            if key:
                sections[key] = '\n'.join(lines)
            key, lines = line.strip('@'), []
        else:
            lines.append(line)
    if key:
        sections[key] = '\n'.join(lines)
    return sections


def parse_sections_re(stdout):
    """Method 2: one regular-expression split on the marker lines gives [before, NAME, body, NAME, body, ...]."""
    parts = re.split(r'^@@(\w+)@@$', stdout, flags=re.M)
    return {k: v.strip('\n') for k, v in zip(parts[1::2], parts[2::2])}


def snapshot_block(sections):
    """Build the [Environment Snapshot] text that agent.py appends to the first prompt (shortened rules)."""
    out = []
    if 'PWD' in sections:
        out.append(f"Working directory: {sections['PWD'].strip()}")
    if 'LS' in sections:
        ls = sections['LS'].strip().split('\n')
        out.append(f'/app contents ({len(ls)} lines):\n' + '\n'.join(ls[:20]) if ls != [''] else '/app contents: (none)')
    for key, title in [('LANG', 'Available languages/tools'), ('PKG', 'Package managers')]:
        if key in sections:
            out.append(f'{title}: ' + '; '.join(l.strip() for l in sections[key].split('\n') if l.strip()))
    if sections.get('MEM', '').strip():
        out.append(f"Memory: {sections['MEM'].strip()}")
    return '[Environment Snapshot]\n' + '\n'.join(out)


def run_bootstrap(app, cwd):
    r = subprocess.run(BOOTSTRAP.format(app=app), shell=True, capture_output=True, text=True, timeout=15, cwd=cwd)
    return r.stdout.strip()


# ---------------------------------------------------------------- the completion state machine of agent.py
def replay(actions, reset_on_command=True):
    """actions: 'cmd' (run commands) or 'done' (call task_complete). Returns the log and the step it ended on.
    agent.py: the first 'done' shows a checklist and sets pending; a second 'done' right after ends the task;
    any command in between sets pending back to False (the line that iteration 4 wanted to remove)."""
    pending, log = False, []
    for step, a in enumerate(actions, 1):
        if a == 'done':
            if pending:
                log.append(f'{step}:done -> END')
                return log, step
            pending = True
            log.append(f'{step}:done -> checklist')
        else:
            if reset_on_command:
                pending = False
            log.append(f'{step}:cmd')
    return log, None


def main():
    log = Log('part4_tb2')
    R = {}
    log('== 1. The environment bootstrap, run on this laptop')
    with tempfile.TemporaryDirectory() as tmp:
        app = os.path.join(tmp, 'app')
        os.makedirs(app)
        for name in ['main.py', 'data.csv', 'README.md']:
            open(os.path.join(app, name), 'w').write('x\n')
        out = run_bootstrap(app, cwd=tmp).replace(os.path.realpath(tmp), '/tmp/demo')
        s1, s2 = parse_sections(out), parse_sections_re(out)
        log('sections found:', list(s1))
        log('Method 1 and Method 2 give the same sections:', {k: v.strip() for k, v in s1.items()} ==
            {k: v.strip() for k, v in s2.items()})
        block = snapshot_block(s1)
        log(block)
        log(f'snapshot length: {len(block)} characters')
        R['with_app'] = dict(sections=list(s1), chars=len(block))

        log('')
        log('-- the same command when the folder does not exist')
        out = run_bootstrap(os.path.join(tmp, 'no-such-dir'), cwd=tmp).replace(os.path.realpath(tmp), '/tmp/demo')
        s = parse_sections(out)
        log('sections found:', list(s))
        log(snapshot_block(s))
        R['no_app'] = dict(sections=list(s))

    log('')
    log('== 2. The completion check: task_complete must be called twice in a row')
    seq = ['cmd', 'cmd', 'done', 'cmd', 'done', 'cmd', 'done', 'done']
    for reset in [True, False]:
        steps, end = replay(seq, reset_on_command=reset)
        name = 'agent.py (reset on command)' if reset else 'iteration 4 idea (no reset)'
        log(f'{name}: ' + ' '.join(steps))
        log(f'   checklists shown: {sum("checklist" in s for s in steps)}, task ends at step {end}')
        R['reset' if reset else 'no_reset'] = dict(steps=steps, end=end)
    save('part4_tb2', R)


if __name__ == '__main__':
    main()
