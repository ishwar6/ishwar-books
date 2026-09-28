// Everything personal lives in site.json (profile, links, projects). It is also read at build time for SEO.
import data from './site.json'

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

export const site = data.site
export const projects: Project[] = data.projects
