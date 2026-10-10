// Render one markdown file with the site's math settings and report KaTeX errors.
// usage (from the repo root): node code/training/check_md.mjs content/books/how-models-are-trained/<part>/<file>.md
import fs from 'node:fs'
import matter from 'gray-matter'
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkRehype from 'remark-rehype'
import rehypeRaw from 'rehype-raw'
import rehypeKatex from 'rehype-katex'
import rehypeStringify from 'rehype-stringify'

for (const f of process.argv.slice(2)) {
  const { content } = matter(fs.readFileSync(f, 'utf8'))
  const html = String(await unified().use(remarkParse).use(remarkGfm).use(remarkMath, { singleDollarTextMath: false })
    .use(remarkRehype, { allowDangerousHtml: true }).use(rehypeRaw).use(rehypeKatex).use(rehypeStringify).process(content))
  const errs = [...html.matchAll(/class="katex-error"[^>]*title="([^"]*)"/g)].map((m) => m[1])
  const lone = content.replace(/```[\s\S]*?```/g, '').replace(/\$\$[\s\S]*?\$\$/g, '').match(/[^\\$]\$[^$]/g) || []
  console.log(`${f}: ${errs.length} KaTeX errors${errs.length ? '\n  ' + errs.join('\n  ') : ''}; ${lone.length} single-$ signs outside math (check they are meant to be literal)`)
}
