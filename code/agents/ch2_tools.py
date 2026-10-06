"""Chapter 2, function calling with a STUBBED model (no API key). Part 1 prints the exact messages of one round trip:
the tool definitions (a real JSON schema), the user turn, the model's structured tool call, the tool result, and the
final answer. Part 2 compares a set of bare endpoint wrappers with a well-described tool set on ten labelled requests. The "model" in
both parts is a stub: a keyword-overlap chooser that stands in for the model's selection step. Its accuracy numbers
show WHY descriptions matter (vague descriptions give the chooser nothing to match), not what a real model scores."""
import json, re, textwrap
from common import Log, save

_log, log2 = Log('ch2_tools'), Log('ch2_tools_select')     # two logs: the round trip, then the selection comparison


def log(s=''):                                             # wrap long JSON lines so the terminal screenshot stays readable
    for line in s.split('\n'):
        for i, part in enumerate(textwrap.wrap(line, 118, subsequent_indent='         ') or ['']):
            _log(part)

# ---- 1. a tool is a name, a description and a JSON schema for its arguments
TOOLS = [
    {'name': 'calendar_find_free_slots',
     'description': 'Find meeting slots when ALL given attendees are free. Use before creating an event. Returns at most 10 slots, '
                    'each as an ISO start time, in the attendees\' time zone. Returns an empty list (not an error) if nobody is free.',
     'input_schema': {'type': 'object',
                      'properties': {'attendee_emails': {'type': 'array', 'items': {'type': 'string', 'format': 'email'},
                                                         'description': 'Email addresses of everyone who must attend.'},
                                     'duration_minutes': {'type': 'integer', 'minimum': 15, 'maximum': 480},
                                     'window_start': {'type': 'string', 'format': 'date', 'description': 'First day to consider, YYYY-MM-DD.'},
                                     'window_end': {'type': 'string', 'format': 'date', 'description': 'Last day to consider, YYYY-MM-DD.'},
                                     'earliest_hour': {'type': 'integer', 'minimum': 0, 'maximum': 23, 'default': 9,
                                                       'description': 'Do not return slots that start before this hour (24h clock).'}},
                      'required': ['attendee_emails', 'duration_minutes', 'window_start', 'window_end']}},
    {'name': 'calendar_create_event',
     'description': 'Create one calendar event and invite the attendees. Idempotent: calling twice with the same idempotency_key returns the same event. Returns the event id.',
     'input_schema': {'type': 'object', 'properties': {'title': {'type': 'string'}, 'start': {'type': 'string', 'format': 'date-time'},
                                                       'duration_minutes': {'type': 'integer'}, 'attendee_emails': {'type': 'array', 'items': {'type': 'string'}},
                                                       'idempotency_key': {'type': 'string'}},
                      'required': ['title', 'start', 'duration_minutes', 'attendee_emails', 'idempotency_key']}},
]


def find_free_slots(attendee_emails, duration_minutes, window_start, window_end, earliest_hour=9):   # a fake calendar
    slots = ['2026-10-12T09:00', '2026-10-12T10:00', '2026-10-13T10:30', '2026-10-14T10:00', '2026-10-15T09:30']
    return {'slots': [s for s in slots if int(s[11:13]) >= earliest_hour], 'time_zone': 'Europe/London'}


def stub_model(messages):
    """Stands in for the LLM. Reads the last message and returns what a real model would be expected to return here."""
    last = messages[-1]
    if isinstance(last['content'], str):                      # a plain user turn: ask for free slots
        return {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'call_01', 'name': 'calendar_find_free_slots',
                 'input': {'attendee_emails': ['priya@example.com', 'tom@example.com'], 'duration_minutes': 45,
                           'window_start': '2026-10-12', 'window_end': '2026-10-16', 'earliest_hour': 10}}]}
    result = json.loads(last['content'][0]['content'])
    return {'role': 'assistant', 'content': [{'type': 'text', 'text': f'Three slots fit (none before 10:00): {", ".join(result["slots"])}. Shall I book the first?'}]}


log('== 1. One function-calling round trip (the model is a stub) ==')
log('REQUEST  system + tools: ' + json.dumps([{'name': t['name'], 'description': t['description'][:60] + '...', 'input_schema': '{...}'} for t in TOOLS]))
log(f'         full schema of {TOOLS[0]["name"]}:\n' + json.dumps(TOOLS[0]['input_schema'], indent=1))
messages = [{'role': 'user', 'content': 'Find 45 minutes for Priya and Tom next week, nothing before 10am.'}]
log('REQUEST  user: ' + json.dumps(messages[0]))
reply = stub_model(messages)
log('REPLY    assistant (stop_reason=tool_use): ' + json.dumps(reply))
call = reply['content'][0]
result = {'calendar_find_free_slots': find_free_slots}[call['name']](**call['input'])      # YOUR code runs the tool
tool_msg = {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': call['id'], 'content': json.dumps(result)}]}
log('REQUEST  tool_result (sent back as the next user turn): ' + json.dumps(tool_msg))
messages += [reply, tool_msg]
final = stub_model(messages)
log('REPLY    assistant (stop_reason=end_turn): ' + json.dumps(final))
log('')

# ---- 2. a vague tool set against a well-described one, on ten labelled requests
BAD = {'get_calendar': 'GET /calendar', 'post_calendar_event': 'POST /calendar/events', 'get_crm_record': 'GET /crm/records/{id}',
       'post_mail': 'POST /mail', 'get_search': 'GET /search?q='}          # five endpoint wrappers, as a generated SDK would name them
GOOD = {'calendar_find_free_slots': 'Find meeting slots when all attendees are free; use before booking. Words: free, slot, availability, when can.',
        'calendar_create_event': 'Create a calendar event and invite attendees; book, schedule, set up a meeting at a known time.',
        'crm_get_customer_context': 'Everything about one customer: account, plan, recent tickets, notes. Use for who is, status of, history of a customer.',
        'email_send': 'Send an email or an invitation message to people. Words: email, mail, message, notify, invite.',
        'docs_search': 'Search internal documents and policies by keyword; find the doc, the policy, the guide about a topic.'}
REQUESTS = [('When are Priya and Tom both free for 45 minutes next week?', 'calendar_find_free_slots'),
            ('Book the design review for Wednesday at 10 with the usual people.', 'calendar_create_event'),
            ('What is the status of customer Acme and their recent tickets?', 'crm_get_customer_context'),
            ('Email the team that the review moved to Wednesday.', 'email_send'),
            ('Find the policy doc on expense limits.', 'docs_search'),
            ('Which slots is the whole team free on Friday?', 'calendar_find_free_slots'),
            ('Schedule a 30 minute call with Sam at 14:00 tomorrow.', 'calendar_create_event'),
            ('Who is the account owner for Globex and what plan are they on?', 'crm_get_customer_context'),
            ('Notify Priya by mail about the room change.', 'email_send'),
            ('Where is the onboarding guide?', 'docs_search')]
MAP_BAD = {'calendar_find_free_slots': 'get_calendar', 'calendar_create_event': 'post_calendar_event', 'crm_get_customer_context': 'get_crm_record',
           'email_send': 'post_mail', 'docs_search': 'get_search'}       # what the endpoint wrappers are "meant" to do
words = lambda s: set(re.findall(r'[a-z]+', s.lower())) - {'the', 'a', 'and', 'for', 'of', 'to', 'on', 'at', 'is', 'with', 'by', 'about', 'their', 'both', 'next', 'week'}


def stub_select(request, tools):
    """The stub's selection step: pick the tool whose name+description shares the most words with the request (ties: first)."""
    scores = {n: len(words(request) & words(n.replace('_', ' ') + ' ' + d)) for n, d in tools.items()}
    best = max(scores.values())
    return next(n for n, s in scores.items() if s == best), scores


log2('== 2. Tool selection by the stub on 10 labelled requests: endpoint wrappers versus described tools ==')
acc = {}
for label, tools, mapper in [('wrapper set', BAD, lambda t: MAP_BAD[t]), ('clear set', GOOD, lambda t: t)]:
    hits = 0
    for req, truth in REQUESTS:
        pick, _ = stub_select(req, tools)
        ok = pick == mapper(truth)
        hits += ok
        log2(f'  [{label}] {"ok " if ok else "BAD"} {req[:52]:<52} -> {pick}')
    acc[label] = hits / len(REQUESTS)
    log2(f'  {label}: {hits}/{len(REQUESTS)} correct')
log2(f'selection accuracy: wrappers {acc["wrapper set"]:.0%}, clear {acc["clear set"]:.0%} (a stub, not a model: it shows the mechanism, not a benchmark)')
save('ch2_tools', {'tools': TOOLS, 'round_trip': messages + [final], 'accuracy': acc, 'bad': BAD, 'good': GOOD})
