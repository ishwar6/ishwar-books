"""Chapter 3: bits per byte of ordinary compressors on the WikiText-2 test text, for comparison with the language models."""
import bz2, gzip, lzma
from common import Log, save
from ch3_common import data

log = Log('ch3_gzip')
text = '\n\n'.join(data('wikitext2_test')['text']).encode('utf-8')
res = {'bytes': len(text)}
for name, fn in [('gzip -9', lambda b: gzip.compress(b, 9)), ('bzip2 -9', lambda b: bz2.compress(b, 9)),
                 ('xz (lzma) -9', lambda b: lzma.compress(b, preset=9))]:
    n = len(fn(text))
    res[name] = 8 * n / len(text)
    log(f'{name:14s} {len(text):,} bytes -> {n:,} bytes   = {8 * n / len(text):.3f} bits per byte')
save('ch3_gzip', res)
