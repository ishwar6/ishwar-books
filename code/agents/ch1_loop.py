"""The SHAPE of an agent loop in 50 lines, with no API key: the "model" below is a stub, a few if-statements that return
the tool call a real model would be expected to make for one fixed on-call scenario. Two fake tools, a loop, a trace.
Nothing here is intelligent; the point is where the model sits, what it sees, and what the loop does with its answer."""
import json
from common import Log

log = Log('ch1_loop')

# ---- two fake tools: an agent acts on the world only through functions like these
def search_logs(service, minutes=15):
    return {'errors': 212, 'top': 'ConnectionPool exhausted (db-primary)', 'first_seen': '14:02', 'deploy': 'v2.31 at 13:58'}

def open_ticket(title, body):
    return {'ticket': 'OPS-4821', 'title': title}

TOOLS = {'search_logs': search_logs, 'open_ticket': open_ticket}

# ---- a stubbed model: a real agent would send `messages` to an LLM here and parse its reply
def model(messages):
    last = messages[-1]
    if last['role'] == 'user':
        return {'thought': 'An alert needs context first. Check recent errors for the service.',
                'tool': 'search_logs', 'args': {'service': 'checkout-api', 'minutes': 15}}
    if last['tool'] == 'search_logs':
        obs = last['content']
        return {'thought': f"{obs['errors']} errors, pool exhaustion, started 4 min after deploy {obs['deploy'].split()[0]}. Likely the deploy.",
                'tool': 'open_ticket', 'args': {'title': 'checkout-api: pool exhausted after v2.31',
                                                'body': 'Roll back v2.31 (errors 14:02, deploy 13:58).'}}
    return {'thought': 'Ticket filed. Report back to the on-call engineer.', 'final': 'Opened OPS-4821 proposing a rollback of v2.31.'}

# ---- the loop: perceive (messages) -> reason (model) -> act (tool) -> observe (result) -> repeat
def run_agent(task, max_steps=5):
    messages = [{'role': 'user', 'content': task}]
    log(f'TASK  {task}')
    for step in range(1, max_steps + 1):
        reply = model(messages)                                  # reason
        log(f'[{step}] THOUGHT  {reply["thought"]}')
        if 'final' in reply:                                     # the model decides it is done
            log(f'[{step}] FINAL    {reply["final"]}')
            return reply['final']
        log(f'[{step}] ACTION   {reply["tool"]}({json.dumps(reply["args"])})')
        result = TOOLS[reply['tool']](**reply['args'])           # act
        log(f'[{step}] OBSERVE  {json.dumps(result)}')
        messages.append({'role': 'tool', 'tool': reply['tool'], 'content': result})   # the observation becomes context
    log('stopped: step budget exhausted')                        # a real loop always has a budget

run_agent('ALERT: checkout-api error rate 12% (threshold 1%). Investigate and propose a fix.')
