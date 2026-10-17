import { Link } from 'react-router-dom'
import { useTitle } from '../lib/hooks'
import { DemoGrid, demos } from '../components/DemoPromo'

export default function Demos() {
  useTitle('Interactive demos', 'Run the models of the papers in your browser: click a photo and SAM cuts out the object, or follow a picture through a Vision Transformer step by step.')
  return (
    <div className="container paper-theme demos-page">
      <header className="demos-hero">
        <span className="eyebrow">The interactive lab</span>
        <h1>Understand it.<br /><span>Then try it yourself.</span></h1>
        <p>Make a prediction. Change an input. See what happens inside the model. Explore the ideas from the papers, one interaction at a time.</p>
        <div className="demos-facts"><span><i />{demos.length} interactive demos</span><span>Runs in your browser</span><span>No setup needed</span></div>
      </header>
      <div className="demos-section-head"><h2>Choose an experiment</h2><span>Real models. Visible steps.</span></div>
      {demos.length ? <DemoGrid /> : <div className="empty">The first demo is being built.</div>}
      <aside className="demos-reading"><div><span className="eyebrow">Go a little deeper</span><h2>There’s a paper behind every demo.</h2><p>Follow the linked explanations for the intuition, equations, and experiments.</p></div><Link to="/papers">Explore the papers <span aria-hidden="true">↗</span></Link></aside>
    </div>
  )
}
