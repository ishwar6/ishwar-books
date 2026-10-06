"""Chapter 2, a toy memory for an agent in about fifty lines: a scratchpad (working memory), a summariser STUB
(no model call; it keeps the first 90 characters of each old message), and a tiny key-value long-term store. We replay
twelve steps of the meeting-scheduling task and print the context size per step with and without compaction.
Token counts use the rule of thumb of 4 characters per token; they are an APPROXIMATION, not a tokenizer."""
from common import Log, save

log = Log('ch2_memory')
tokens = lambda s: len(s) // 4                          # approximation: about 4 characters per token


class Memory:
    def __init__(self, compact_every=4, keep_last=2):
        self.history, self.scratch, self.store = [], [], {}  # short-term, working, long-term
        self.compact_every, self.keep_last = compact_every, keep_last

    def remember(self, key, value):                     # long-term: survives compaction and future runs
        self.store[key] = value

    def recall(self, query):                            # the real thing would be a vector or full-text search
        return {k: v for k, v in self.store.items() if any(w in k for w in query.split())}

    def note(self, line):                               # working memory: the running plan, written by the agent
        self.scratch.append(line)

    def add(self, role, content, step):
        self.history.append({'role': role, 'content': content})
        if step % self.compact_every == 0:              # compaction: old turns become one short summary
            old, recent = self.history[:-self.keep_last], self.history[-self.keep_last:]
            summary = ' '.join(m['content'].split('. ')[0][:90].rstrip('.') + '.' for m in old if m['role'] != 'summary')
            prior = [m['content'] for m in old if m['role'] == 'summary']
            self.history = [{'role': 'summary', 'content': ' '.join(prior + [summary])}] + recent

    def context(self, query):                           # what the model would see on this turn
        parts = ['SYSTEM: You schedule meetings. Tools: find_free_slots, create_event, send_email.']
        parts += [f'MEMORY {k}: {v}' for k, v in self.recall(query).items()]
        parts += ['PLAN: ' + ' | '.join(self.scratch)]
        parts += [f'{m["role"].upper()}: {m["content"]}' for m in self.history]
        return '\n'.join(parts)


# ---- twelve steps of the meeting task, as (role, text, fact-to-store or None, note-to-write or None)
STEPS = [
    ('user', 'Set up a 45 minute design review with Priya and Tom next week. I prefer mornings but never before 10am. Send invites.', ('preference meetings', 'never before 10:00; prefers mornings'), 'goal: 45-min review, Priya + Tom, next week'),
    ('tool', 'find_free_slots -> 14 shared slots: Mon 10:00, Mon 11:00, Mon 14:00, Tue 09:00, Tue 10:30, Tue 15:00, Wed 10:00, Wed 11:30, Wed 16:00, Thu 09:30, Thu 10:00, Thu 13:00, Fri 10:00, Fri 11:00. Each slot is 30 minutes long; adjacent slots can be combined for longer meetings.', None, 'candidates: Mon 10, Wed 10, Thu 10, Fri 10 (>= 10:00, morning)'),
    ('assistant', 'Tue 09:00 and Thu 09:30 are too early. Mon 10:00 fits the preference. I will check the room next.', None, None),
    ('tool', 'find_rooms -> Room 4B free Mon 10:00-11:00 (capacity 6, screen). Room 2A busy. Room 7C free but capacity 2.', None, 'room: 4B'),
    ('assistant', 'Room 4B works. Creating the event for Mon 10:00-10:45 with Priya and Tom in 4B.', None, None),
    ('tool', 'create_event -> ok: event id EVT-1187, Mon 10:00-10:45, attendees priya@, tom@, room 4B, title Design review.', ('event design review', 'EVT-1187 Mon 10:00-10:45 room 4B'), 'done: event created'),
    ('assistant', 'Event created. Sending invites to both attendees with the agenda line from the request.', None, None),
    ('tool', 'send_email -> ok: 2 invitations sent (priya@, tom@). Error: Tom has an auto-reply saying he is out on Monday and back Tuesday.', None, 'problem: Tom out Monday'),
    ('assistant', 'Tom is out Monday. I must move the meeting. Checking the stored preference and the remaining morning slots.', None, None),
    ('tool', 'find_free_slots -> remaining shared morning slots not before 10:00: Wed 10:00, Thu 10:00, Fri 10:00; room 4B free Wed 10:00 and Fri 10:00.', None, 'candidates: Wed 10, Fri 10'),
    ('tool', 'update_event -> ok: EVT-1187 moved to Wed 10:00-10:45, room 4B; updated invitations sent.', ('event design review', 'EVT-1187 Wed 10:00-10:45 room 4B'), 'done: moved to Wed'),
    ('assistant', 'Done. The design review is Wed 10:00-10:45 in 4B; both attendees have the updated invitation.', None, None),
]

rows = []
for label, mem in [('without compaction', Memory(compact_every=10 ** 9)), ('with compaction', Memory(compact_every=4))]:
    log(f'== {label}: context size per step (approximate tokens, 4 chars per token) ==')
    sizes = []
    for i, (role, txt, fact, note) in enumerate(STEPS, 1):
        if fact:
            mem.remember(*fact)
        if note:
            mem.note(note)
        mem.add(role, txt, i)
        ctx = mem.context('meetings design review')
        sizes.append(tokens(ctx))
        log(f'  step {i:>2}: {tokens(ctx):>4} tokens, {len(mem.history):>2} messages in history')
    log(f'  total over 12 steps: {sum(sizes):,} tokens; largest {max(sizes)}')
    rows.append({'label': label, 'sizes': sizes, 'total': sum(sizes)})
    log('')

log('== what the compacted agent sees at step 12 ==')
import textwrap
for line in mem.context('meetings design review').split('\n'):
    log('\n'.join(textwrap.wrap(line, 118, subsequent_indent='    ')))
log('')
log(f'recall("meetings") after compaction -> {mem.recall("meetings")}')
log('the user turn that stated the 10:00 rule was summarised away at step 4; the rule survives in the long-term store.')
save('ch2_memory', {'rows': rows, 'final_context': mem.context('meetings design review'), 'store': mem.store, 'scratch': mem.scratch})
