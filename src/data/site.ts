// Everything personal lives here. Edit this file to update the site's profile, links and projects.
export const site = {
  name: 'Ishwar Jangid',
  firstName: 'Ishwar',
  tagline: 'systems, machine learning, and GPUs. learning in public.',
  location: 'Delhi, India',
  email: 'ishwarjangid116@gmail.com',
  bio: [
    'I am a software engineer who builds production AI systems: agents, retrieval pipelines and the data platforms underneath them.',
    'I learn by writing things down properly. This site is where those notes become long-form, first-principles books on GPU programming, training LLMs from scratch, retrieval-augmented generation and ML systems, each with code you can run and numbers you can measure.',
  ],
  links: {
    github: 'https://github.com/ishwar6',
    youtube: 'https://www.youtube.com/c/IshwarJangid',
    x: 'https://x.com/Ishwaraiml',
  },
}

export type Project = {
  name: string
  tagline: string
  description: string
  highlights: string[]
  tech: string[]
  /** Optional small label above the name. */
  status?: string
  /** Optional public link; private projects have none. */
  url?: string
}

// Add new projects to the top of this list.
export const projects: Project[] = [
  {
    name: 'Manim Studio',
    tagline: 'An explainer-video pipeline: animation, voice and code generation in one place',
    description:
      'A local platform for making 3Blue1Brown-style explainer videos. A video is written as a script of short beats; each beat is narrated in my own voice, animated in Manim, and synced so the motion lands on the spoken word. The first series explains vector databases in nine episodes, from "meaning as geometry" to HNSW, filtering, hybrid search and quantization.',
    highlights: [
      'Beat-by-beat script editor: narration, visual spec, camera moves, on-screen text and transitions for every beat, with full revision history.',
      'Voice recorded per beat in the browser, then mastered automatically (rumble removal, denoising, compression, loudness normalisation) without changing its length.',
      'Audio sync: sync marks in the narration become animation cues, timed from Whisper word timestamps on real recordings or exact character timings from text-to-speech.',
      'AI-assisted Python generation: Claude Code writes the Manim scene code from each beat spec, renders it, checks the frames, and snapshots every build so any change can be undone.',
      'A 3Blue1Brown look: Computer Modern math with a fixed colour per symbol, a mascot that reacts in comic bubbles, and sound effects placed from the script.',
    ],
    tech: ['Python', 'Manim', 'FastAPI', 'React', 'PostgreSQL', 'ffmpeg', 'Whisper', 'Claude Code'],
  },
]
