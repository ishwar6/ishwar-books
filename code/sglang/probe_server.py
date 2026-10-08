"""Two-request streaming smoke test, NOT a load benchmark. Standard library only.
Usage: python probe_server.py --url http://127.0.0.1:30000 --model comparison-model --output /tmp/probe.json
No API calls occur unless this script is explicitly run against a server.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import time
import urllib.request

def consume_stream(lines, start, clock=time.perf_counter):
    first=None; chunks=[]; usage=None; done=False
    for line in lines:
        line=line.decode('utf-8').strip()
        if not line.startswith('data:'): continue
        payload=line[5:].strip()
        if payload=='[DONE]': done=True; break
        obj=json.loads(payload)
        if 'error' in obj: raise RuntimeError(str(obj['error']))
        for choice in obj.get('choices',[]):
            text=choice.get('text','')
            if text:
                if first is None: first=clock()
                chunks.append(text)
        if obj.get('usage') is not None: usage=obj['usage']
    end=clock()
    if not done: raise RuntimeError('Stream ended without [DONE]; result is incomplete')
    if first is None: raise RuntimeError('No non-empty completion text received')
    return {'observed_first_text_ms':(first-start)*1000,'total_stream_ms':(end-start)*1000,
            'usage':usage,'text':''.join(chunks),'text_chunks':len(chunks)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',required=True); p.add_argument('--model',required=True)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    prefix='Reference handbook.\n'+''.join(f'Policy {i}: goods can be returned within thirty days when unused.\n' for i in range(80))
    result={'kind':'two-request smoke test; not a throughput benchmark','url':args.url,'model':args.model,
            'python':platform.python_version(),'requests':[],'cache_state':'unknown at start; no cache-hit claim'}
    for question in ['What is the return window?','Must the goods be unused?']:
        body={'model':args.model,'prompt':prefix+'\nQuestion: '+question+'\nAnswer:',
              'max_tokens':32,'temperature':0,'stream':True,'stream_options':{'include_usage':True}}
        headers={'Content-Type':'application/json'}
        if os.environ.get('INFERENCE_API_KEY'): headers['Authorization']='Bearer '+os.environ['INFERENCE_API_KEY']
        req=urllib.request.Request(args.url.rstrip('/')+'/v1/completions',data=json.dumps(body).encode(),headers=headers)
        start=time.perf_counter()
        with urllib.request.urlopen(req,timeout=180) as response:
            observation=consume_stream(response,start)
        result['requests'].append({'question':question,'request':body,'observation':observation})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Saved smoke test:',args.output)
if __name__=='__main__': main()
