import { videos } from '../lib/content'
import { useTitle } from '../lib/hooks'
import { VideoCard } from '../components/Cards'

export default function Videos() {
  useTitle('Videos', 'Animated explainers on vector databases, search and AI systems, built from first principles and measured on real data.')
  return (
    <div className="container">
      <header className="page-head writings-head">
        <span className="eyebrow">Videos</span>
        <h1>Animated <span className="grad">explainers</span></h1>
        <p>Deep dives on how AI systems really work, animated from first principles and measured on real data. Each video has a written companion with the key ideas, the numbers and the full transcript.</p>
      </header>
      {videos.length ? (
        <>
          <VideoCard video={videos[0]} featured />
          {videos.length > 1 && <div className="video-grid" style={{ marginTop: 20 }}>{videos.slice(1).map((v) => <VideoCard key={v.route} video={v} />)}</div>}
        </>
      ) : (
        <div className="empty">The first video is on its way.</div>
      )}
    </div>
  )
}
