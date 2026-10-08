"""Build theme-aware SVG teaching figures directly from saved experiment results."""
import json
from pathlib import Path
from html import escape
R=Path(__file__).resolve().parent
D=json.loads((R/'results/experiments.json').read_text())
def text(x,y,s,cls='t-note',anchor='middle'):
 return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{escape(str(s))}</text>'
def rect(x,y,w,h,cls='box-1'):
 return f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>'
def line(x,y,a,b): return f'<line class="edge" x1="{x}" y1="{y}" x2="{a}" y2="{b}"/>'
def svg(h,body,label): return f'<svg viewBox="0 0 760 {h}" role="img" aria-label="{escape(label)}">{body}</svg>'
f={}
b=rect(220,20,320,55)+text(380,43,'Shared prefix: policy + document')+text(380,62,'one set of KV tensors','t-tick')
for x,label in [(20,'Summarise'),(280,'Extract risks'),(540,'Draft reply')]:
 b+=line(380,75,x+100,120)+rect(x,120,200,62,'box-2')+text(x+100,145,label)+text(x+100,165,'compute this suffix','t-tick')
b+=text(380,220,'Tree on CPU: token paths and pointers. KV tensors: device memory.','t-tick')
f['tree']=svg(245,b,'Three requests share a cached policy and document, then branch into different suffixes.')
b=''
rows=[('No reuse',86016),('Radix index',D['aligned']['radix']['computed_tokens']),('Block-hash index',D['aligned']['blocks16']['computed_tokens'])]
for i,(label,val) in enumerate(rows):
 y=25+i*65;b+=text(180,y+24,label,anchor='end')+rect(200,y,450*val/86016,36,'box-2' if i==0 else 'box-1')+text(210+450*val/86016,y+24,f'{val:,}',anchor='start')
b+=text(380,235,'64 synthetic prompts; computed prompt tokens, not milliseconds','t-tick')
f['work']=svg(255,b,'Both teaching cache indexes compute 6,144 tokens versus 86,016 without reuse.')
b=''
for row,base in enumerate([25,130]):
 b+=text(15,base+15,'Serial' if row==0 else 'Overlap',anchor='start')
 for i in range(5):
  start=110+i*(100 if row==0 else 80)
  b+=rect(start,base,20,25,'box-2')+rect(start+20,base+32,80,25,'box-1')
 b+=text(15,base+48,'GPU',cls='t-tick',anchor='start')
b+=text(380,240,'Illustrative: CPU preparation 2 ms, GPU work 8 ms; each unit is 1 ms.','t-tick')
b+=text(380,263,'After startup, the CPU prepares the next batch while the GPU runs.','t-tick')
f['overlap']=svg(285,b,'CPU scheduling overlaps GPU execution. Ideal steady state changes from 10 milliseconds to 8 milliseconds per batch.')
b=''
for x,t,sub in [(20,'GPU memory','ready to reuse'),(280,'Host memory','copy back first'),(540,'Storage','fetch + copy first')]:
 b+=rect(x,30,200,75)+text(x+100,60,t)+text(x+100,85,sub,'t-tick')
b+=line(220,68,280,68)+line(480,68,540,68)+text(380,155,'A larger cache helps only when retrieving KV beats recomputing it.','t-tick')
f['tiers']=svg(180,b,'Hierarchical KV caching trades slower transfers for greater capacity.')
(R/'results/figures.json').write_text(json.dumps(f,indent=2)+'\n')
print('Wrote',len(f),'figures')
