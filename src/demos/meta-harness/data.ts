/* Loading for the Meta-Harness demo. Every file is made by code/papers/meta-harness/demo/:
   paper.json (the paper's numbers, checked against the PDF text by export_paper.py) and run/ (one real search run of
   our small Meta-Harness from code/papers/meta-harness/scratch/, exported by export_run.py). */

export const BASE = import.meta.env.BASE_URL + 'demos/meta-harness/'

const cache = new Map<string, Promise<unknown>>()
function once<T>(key: string, f: () => Promise<T>): Promise<T> {
  if (!cache.has(key)) cache.set(key, f().catch((e) => { cache.delete(key); throw e }))
  return cache.get(key) as Promise<T>
}
async function json<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path)
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`)
  return r.json() as Promise<T>
}

/* ---------------- the paper's numbers ---------------- */

type Src = { source: string; page: number; what?: string }
export type Paper = {
  table1: Src & { unit: string; rows: { method: string; history: string; log: string; mtok: number }[] }
  table3: Src & { rows: { id: 'scores' | 'summary' | 'full'; method: string; scores: boolean; code: boolean; summary: boolean | null
    traces: boolean; median: number; best: number; above_zs: number }[] }
  table4: Src & { rows: { method: string; median: number; best: number }[] }
  table8: Src & { files_median: number; files_range: string; breakdown: { kind: string; pct: number }[] }
  table2: Src & { rows: { harness: string; uspto: number; s2d: number; law: number; avg: number; ctx_ktok: number }[] }
  table9: Src & { rows: { variant: string; uspto: number; s2d: number; law: number; avg: number; ctx_kchar: number }[] }
  figure3: Src & { baselines: { harness: string; table2: string; x_kchar_from_plot: number }[] }
  table6: Src & { models: { id: string; name: string; short: string }[]
    rows: { method: string; acc: number[]; delta?: number[]; avg: number; avg_delta?: number }[] }
  table7: Src & { groups: { model: string; rows: { harness: string; auto: boolean; pass: number }[] }[]; notes: string[] }
  appendixA: Src & { baseline: number; tries: { name: string; pass: number; change: number }[] }
}

export const loadPaper = () => once('paper', () => json<Paper>('paper.json'))

/* ---------------- our search run (export_run.py) ---------------- */

export type Call = { prompt: string; reply: string }
export type TraceFull = { id: number; text: string; gold: string; pred: string | null; ok: boolean; calls: Call[]; chars: number; error?: string }
export type SearchScores = {
  accuracy: number; correct: number; n: number; ctx_chars: number; calls_per_message: number; invalid: number; crashes: number
  accuracy_by_label: Record<string, string>; top_confusions: Record<string, number>
}
export type Cand = {
  folder: string; iter: number; seed: boolean; frontier: boolean; parent: string | null; proposal: string; code: string
  files: { path: string; bytes: number }[]
  search: SearchScores
  test: { acc: number; correct: number; n: number; ctx_chars: number; ok: number[] } | null
  ok: number[]; pred: (string | null)[]
}
export type Rejected = { folder: string; iter: number; parent: string | null; proposal: string; error: string; code: string }
export type RunMeta = {
  name: string; proposer: string; iterations: number; k: number; task_model: string; device: string; labels: string[]
  sizes: { memory: number; search: number; test: number }
  stats: { proposed: number; rejected: number; duplicates: number; evaluated: number }
  selected: string; minutes_search: number; minutes_total: number; task_model_calls: number; proposer_model?: string
}
export type Run = { name: string; meta: RunMeta; candidates: Cand[]; rejected: Rejected[]; messages: { text: string; gold: string }[]; log: string }
export type RunIndex = { runs: { file: string; name: string; proposer: string; candidates: number }[] }

export const loadRunIndex = () => once('run-index', () => json<RunIndex>('run/index.json'))
export const loadRun = (file: string) => once('run-' + file, () => json<Run>('run/' + file))
export const loadTraces = (run: string, folder: string) => once(`tr-${run}-${folder}`, () => json<TraceFull[]>(`run/${run}/${folder}.json`))

/** Human words for a proposer id. */
export const proposerName = (p: string) => ({ scripted: 'scripted proposer (rules over a design space)', 'local-llm': 'local LLM proposer',
  api: 'chat API proposer', 'agent-cli': 'coding-agent proposer' } as Record<string, string>)[p] ?? p
