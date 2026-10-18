/* The Meta-Harness demo (/papers/meta-harness/demo): three parts. No model runs in the page; this is about the search loop.
   1. A real search run of our small Meta-Harness (code/papers/meta-harness/scratch/), exported to JSON (Run.tsx).
   2. What the proposer sees: one failure as scores, a summary, or full traces, with the paper's ablation (Proposer.tsx).
   3. The paper's own numbers: TerminalBench-2, the Pareto frontier, math across models (Results.tsx). */
import { lazy, Suspense, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import DemoTabs from '../../components/DemoTabs'
import { BG, P1, P2 } from './links'
import { Busy } from './ui'
import './meta-harness-demo.css'

const Run = lazy(() => import('./Run'))
const Proposer = lazy(() => import('./Proposer'))
const Results = lazy(() => import('./Results'))

function Part({ id, n, title, children }: { id: string; n: string; title: string; children: ReactNode }) {
  return (
    <section className="meta-harness-demo-part" id={id} aria-labelledby={`${id}-h`}>
      <h2 id={`${id}-h`}><span>{n}</span>{title}</h2>
      <Suspense fallback={<Busy text="Loading…" />}>{children}</Suspense>
    </section>
  )
}

export default function Demo() {
  return (
    <div className="meta-harness-demo">
      <p className="meta-harness-demo-intro">
        A <b>harness</b> is the code around a language model: what it stores, what it looks up, and what it puts in each prompt.
        Meta-Harness lets a coding agent, the <b>proposer</b>, write new harnesses. Each one is scored, and its code, score and
        traces go into a folder the proposer can read next time. That is the <Link to={`${P2}`}>loop of Figure 2</Link>. Here you
        can open such a folder, choose what a proposer may read, and explore the <Link to={P1}>paper's results</Link>. New to
        harnesses? Start with <Link to={BG}>the Background page</Link>.
      </p>
      <DemoTabs id="meta-harness" label="Three ways into Meta-Harness" tabs={[
        { id: 'run', label: 'Inside a search run', content: (
          <Part id="mh-run" n="1" title="Inside a search run"><Run /></Part>) },
        { id: 'proposer', label: 'What the proposer sees', content: (
          <Part id="mh-proposer" n="2" title="What the proposer sees"><Proposer /></Part>) },
        { id: 'results', label: "The paper's results", content: (
          <Part id="mh-results" n="3" title="The paper's results"><Results /></Part>) },
      ]} />
    </div>
  )
}
