"""Part 8: watch the memory footprint of a running training process (what Activity Monitor shows as Memory; on Apple
silicon it includes the mps buffers, which the resident size of psutil does not). One line per minute, next to the
step the training log has reached. No model is loaded.  Usage: python r1_part8_memwatch.py <pid> <tag>"""
import os, re, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

pid, tag = int(sys.argv[1]), sys.argv[2]
log = common.Log(f'r1_part8_memory_{tag}')
train_log = os.path.join(common.RESULTS, f'r1_part8_train_{tag}_stdout.txt')
t0 = time.time()
while True:
    try:
        os.kill(pid, 0)
    except OSError:
        log('training process ended'); break
    out = subprocess.run(['footprint', '-p', str(pid)], capture_output=True, text=True).stdout
    m = re.search(r'Footprint: ([\d.]+) (GB|MB)', out)
    steps = [l for l in open(train_log) if l.startswith('step')] if os.path.exists(train_log) else []
    last = steps[-1].split()[1] if steps else '0'
    if m:
        gb = float(m.group(1)) / (1 if m.group(2) == 'GB' else 1024)
        log(f'minute {(time.time() - t0) / 60:5.1f}  after step {last:>4}  footprint {gb:5.1f} GB')
    time.sleep(60)
