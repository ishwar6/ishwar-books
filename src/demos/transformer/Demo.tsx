/* The Transformer demo (/papers/transformer/demo): three parts.
   1. Our from-scratch toy Transformer, live in the browser (Toy.tsx).
   2. The released translator opus-mt-en-de, precomputed in Python (Marian.tsx).
   3. Equation 1 by hand (Calc.tsx). */
import { lazy, Suspense, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import DemoTabs from '../../components/DemoTabs'
import Toy from './Toy'
import { P2, P3, BG } from './links'
import './transformer-demo.css'

const Marian = lazy(() => import('./Marian'))
const Calc = lazy(() => import('./Calc'))

function Part({ id, n, title, children }: { id: string; n: string; title: string; children: ReactNode }) {
  return (
    <section className="transformer-demo-part" id={id} aria-labelledby={`${id}-h`}>
      <h2 id={`${id}-h`}><span>{n}</span>{title}</h2>
      {children}
    </section>
  )
}

export default function Demo() {
  return (
    <div className="transformer-demo">
      <p className="transformer-demo-intro">
        A Transformer reads a whole input at once and writes its output one token at a time. Its encoder turns the input
        into vectors, with attention between all the tokens; its decoder writes, looking at what it wrote and at the input.
        The <Link to={`${P2}#figure-1-the-whole-model-in-one-picture`}>whole model</Link> is one picture in the paper. Here
        you can follow every step of it with real numbers. New to translation models? Start with <Link to={BG}>the Background page</Link>.
      </p>
      <DemoTabs id="transformer" label="Three ways into the Transformer" tabs={[
        { id: 'live', label: 'A date, live', content: (
          <Part id="tx-live" n="1" title="Follow a date through our Transformer">
            <Toy />
          </Part>) },
        { id: 'translator', label: 'A real translator', content: (
          <Part id="tx-real" n="2" title="Inside a real translator, English to German">
            <Suspense fallback={<div className="transformer-demo-busy">Loading…</div>}><Marian /></Suspense>
          </Part>) },
        { id: 'calculator', label: 'Attention by hand', content: (
          <Part id="tx-calc" n="3" title="Attention by hand: Equation 1">
            <p>
              One attention step is <Link to={`${P3}#equation-1-step-by-step`}>Equation 1</Link>: softmax(Q Kᵀ / √d_k) V. Each token
              has a query (what it looks for), a key (the label that queries are compared with) and a value (what it gives back).
              Here each vector has d_k = 2 numbers, so you can see them as arrows. Change any number, or drag an arrow, and every
              table below updates.
            </p>
            <Suspense fallback={<div className="transformer-demo-busy">Loading…</div>}><Calc /></Suspense>
          </Part>) },
      ]} />
    </div>
  )
}
