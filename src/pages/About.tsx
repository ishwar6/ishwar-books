import { site } from '../data/site'
import { useTitle } from '../lib/hooks'

export default function About() {
  useTitle('About')
  return (
    <div className="container">
      <header className="page-head">
        <span className="eyebrow">About</span>
        <h1>Hi, I am {site.name}</h1>
      </header>
      <div className="about-grid">
        <div className="prose">
          {site.bio.map((p) => <p key={p}>{p}</p>)}
          <h2>What I write about</h2>
          <ul>
            <li><strong>GPU programming</strong>: CUDA, Triton, tensor cores and the performance model behind fast kernels.</li>
            <li><strong>Training language models</strong>: from the tokenizer to pretraining, fine-tuning and RL.</li>
            <li><strong>Retrieval and agents</strong>: RAG systems you can measure, and agent graphs you can run in production.</li>
          </ul>
          <h2>Get in touch</h2>
          <p>The best way to reach me is email at <a href={`mailto:${site.email}`}>{site.email}</a>. I am always happy to talk about systems, ML and teaching.</p>
        </div>
        <dl className="about-card">
          <dt>Email</dt><dd><a href={`mailto:${site.email}`}>{site.email}</a></dd>
          <dt>Based in</dt><dd>{site.location}</dd>
          <dt>GitHub</dt><dd><a href={site.links.github} target="_blank" rel="noreferrer">github.com/ishwar6</a></dd>
          <dt>YouTube</dt><dd><a href={site.links.youtube} target="_blank" rel="noreferrer">Ishwar Jangid</a></dd>
          <dt>X</dt><dd><a href={site.links.x} target="_blank" rel="noreferrer">@Ishwaraiml</a></dd>
        </dl>
      </div>
    </div>
  )
}
