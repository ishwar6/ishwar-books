"""Render the causal-prefix sentence from the pinned SGLang paper.
Usage: python paper_shots.py /path/to/2312.07104v2.pdf
Provenance stays in results/paper_shots.json; the article explains the finding.
"""
import hashlib,json,sys
from pathlib import Path
import pymupdf as fitz
ROOT=Path(__file__).resolve().parents[2]
source=Path(sys.argv[1]); doc=fitz.open(source); page=doc[3]
phrase='KV cache computation depends only on prefix tokens.'
hits=page.search_for(phrase)
assert len(hits)==1,(phrase,hits)
box=hits[0]
clip=fitz.Rect(box.x0-1,box.y0-.7,box.x1+1,box.y1+.7)
ann=page.add_highlight_annot(box); ann.set_colors(stroke=(1,.88,.25)); ann.update()
pix=page.get_pixmap(matrix=fitz.Matrix(5,5),clip=clip,alpha=False)
out=ROOT/'public/img/sglang'; out.mkdir(parents=True,exist_ok=True)
pix.save(out/'paper-prefix.png')
manifest={'url':'https://arxiv.org/pdf/2312.07104v2','sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'crops':[{'page':4,'phrase':phrase,'crop':list(clip)}],
          'note':'One source sentence, rendered from the PDF with a highlight; no reconstructed text or added image labels.'}
(ROOT/'code/sglang/results/paper_shots.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Wrote paper-prefix.png')
