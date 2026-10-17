import { execFileSync } from 'node:child_process'

/** Source commit dates, never the build clock. A shallow/archive build can omit unknown dates. */
export function contentDates(root) {
  const dates = new Map()
  try {
    const log = execFileSync('git', ['-c', 'core.quotepath=false', 'log', '--format=@@date:%cs', '--name-only', '--no-renames', '--', 'content', 'src/data', 'src/pages', 'src/demos'], { cwd: root, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 })
    let date
    for (const line of log.split('\n')) {
      if (line.startsWith('@@date:')) date = line.slice(7)
      else if (line && date && !dates.has(line)) dates.set(line, date)
    }
  } catch { console.warn('Content history unavailable; using explicit dates where supplied.') }
  return dates
}

export const latestDate = (...dates) => dates.flat().filter(Boolean).sort().at(-1)
