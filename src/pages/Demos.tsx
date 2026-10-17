import { Link } from 'react-router-dom'
import { useTitle } from '../lib/hooks'
import { DemoGrid, demos } from '../components/DemoPromo'

export default function Demos() {
  useTitle('Interactive demos', 'Run the models of the papers in your browser: click a photo and SAM cuts out the object, or follow a picture through a Vision Transformer step by step.')
  return (
    <div className="container paper-theme">
      <header className="page-head papers-head">
        <span className="eyebrow">Interactive demos</span>
        <h1>Run the paper, <span className="grad">in your browser</span></h1>
        <p>
          Each demo runs the real model of a paper on your clicks and shows every step on the way: the shapes, the numbers
          and the pictures. Nothing is sent to a server. Each step links to the part of the series that explains it.
        </p>
      </header>
      {demos.length ? <DemoGrid /> : <div className="empty">The first demo is being built.</div>}
      <p className="demos-more">The full explanations are in <Link to="/papers">Research papers</Link>.</p>
    </div>
  )
}
