"""Render genuine small excerpts from the pinned SGLang paper; do not retype paper text.
Usage: python paper_shots.py /path/to/2312.07104v2.pdf
Requires pymupdf and pillow. Produces an explicitly labelled two-crop panel.
"""
import hashlib,json,sys
from pathlib import Path
import pymupdf as fitz
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
source=Path(sys.argv[1]); doc=fitz.open(source); page=doc[3]
phrases=['Efficient KV Cache Reuse with RadixAttention','KV cache computation depends only on prefix tokens.']
font_path='/System/Library/Fonts/Supplemental/Arial.ttf'
try: font=ImageFont.truetype(font_path,24)
except OSError: font=ImageFont.load_default(size=24)
canvas=Image.new('RGB',(1200,310),'#f7f8fa'); draw=ImageDraw.Draw(canvas)
draw.text((30,20),'SGLang paper · arXiv:2312.07104v2 · page 4 · two separate crops',fill='#344054',font=font)
records=[]
for phrase,y in zip(phrases,[82,184]):
 hits=page.search_for(phrase); assert len(hits)==1,(phrase,hits)
 box=hits[0]
 # Exact text line, with just 0.7pt vertical padding to exclude neighbouring lines.
 clip=fitz.Rect(box.x0-1,box.y0-.7,box.x1+1,box.y1+.7)
 if y==184:
  ann=page.add_highlight_annot(box); ann.set_colors(stroke=(1,.88,.25)); ann.update()
 pix=page.get_pixmap(matrix=fitz.Matrix(4,4),clip=clip,alpha=False)
 im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
 canvas.paste(im,(30,y)); records.append({'page':4,'phrase':phrase,'crop':list(clip)})
 draw.text((30,y+58),'Section heading' if y==82 else 'The causal-prefix condition (highlight added)',fill='#667085',font=font)
out=ROOT/'public/img/sglang'; out.mkdir(parents=True,exist_ok=True)
canvas.save(out/'paper-prefix.png')
manifest={'url':'https://arxiv.org/pdf/2312.07104v2','sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'crops':records,'note':'Cropped source pixels, no text reconstruction. Two crops separated and labelled; highlight added.'}
(ROOT/'code/sglang/results/paper_shots.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Wrote paper-prefix.png')
