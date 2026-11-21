"""Run the Part 6 experiments under ONE heavy-job lock: r1_part6.py (the loss, two ways; base vs distilled),
then r1_part6_eval.py (accuracy and length), then, with `--tiny`, r1_part6_tiny.py (a tiny distillation run)."""
import sys
from common import heavy
import r1_part6, r1_part6_eval, r1_part6_tiny

if __name__ == '__main__':
    with heavy('part6'):
        if '--tiny-only' not in sys.argv:
            r1_part6.main(lock=False)
            r1_part6_eval.main(lock=False)
        if '--tiny' in sys.argv or '--tiny-only' in sys.argv:
            r1_part6_tiny.main(lock=False)
