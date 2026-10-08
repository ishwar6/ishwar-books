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
# Follow one question through the engine before introducing cache structures.
b=text(30,25,'INPUT',cls='t-tick',anchor='start')+rect(20,40,720,52)+text(380,73,'Instructions + handbook + question')
b+=line(380,92,380,118)+rect(100,118,560,60,'box-2')+text(380,144,'PREFILL: process the supplied text')+text(380,165,'Build KV state and select the first answer token','t-tick')
b+=line(380,178,380,204)+rect(100,204,560,60,'box-1')+text(380,230,'DECODE: continue the answer')+text(380,251,'Read saved KV, generate the next token, repeat','t-tick')
b+=line(380,264,380,291)+text(380,315,'OUTPUT: the answer streams back to the user')
f['request']=svg(345,b,'Instructions, handbook and question enter prefill, which builds KV and predicts the first output token. Decode continues the answer.')
# Concrete edge split, using exactly the IDs in walkthrough.py.
b=text(25,25,'1. After the first request',anchor='start')
b+=rect(90,48,580,50)+text(380,79,'10 11 12 13 14 15 20 21')
b+=text(380,123,'One stored path: eight computed token positions','t-tick')
b+=text(25,177,'2. The second request matches six IDs, then branches',anchor='start')
b+=rect(210,198,340,52)+text(380,230,'10 11 12 13 14 15')
b+=line(380,250,190,293)+line(380,250,570,293)
b+=rect(90,293,200,48,'box-1')+text(190,323,'20 21')
b+=rect(470,293,200,48,'box-2')+text(570,323,'30 31')
b+=text(190,367,'First question: already stored','t-tick')+text(570,367,'Second question: compute now','t-tick')
b+=text(380,405,'The shared path refers to one reusable set of KV states.','t-tick')
f['radix_steps']=svg(430,b,'First request stores eight tokens. The second shares token IDs 10 through 15 and splits into question branches 20 21 and 30 31.')
b=''
rows=[('No reuse',86016),('Radix index',D['aligned']['radix']['computed_tokens']),('Block-hash index',D['aligned']['blocks16']['computed_tokens'])]
for i,(label,val) in enumerate(rows):
 y=25+i*65;b+=text(180,y+24,label,anchor='end')+rect(200,y,450*val/86016,36,'box-2' if i==0 else 'box-1')+text(210+450*val/86016,y+24,f'{val:,}',anchor='start')
b+=text(380,235,'Prompt token positions computed across 64 requests','t-tick')
f['work']=svg(255,b,'Both teaching cache indexes compute 6,144 tokens versus 86,016 without reuse.')
b=''
for row,base in enumerate([25,130]):
 b+=text(15,base-7,'Serial' if row==0 else 'Overlap',anchor='start')+text(15,base+18,'CPU',cls='t-tick',anchor='start')
 for i in range(5):
  start=110+i*(100 if row==0 else 80)
  b+=rect(start,base,20,25,'box-2')+rect(start+20,base+32,80,25,'box-1')
 b+=text(15,base+48,'GPU',cls='t-tick',anchor='start')
b+=text(380,240,'Illustrative: CPU preparation 2 ms, GPU work 8 ms; each unit is 1 ms.','t-tick')
b+=text(380,263,'After startup, the CPU prepares the next batch while the GPU runs.','t-tick')
f['overlap']=svg(285,b,'CPU scheduling overlaps GPU execution. Ideal steady state changes from 10 milliseconds to 8 milliseconds per batch.')
(R/'results/figures.json').write_text(json.dumps(f,indent=2)+'\n')
print('Wrote',len(f),'figures')
