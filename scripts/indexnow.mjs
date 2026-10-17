// Tell search engines that use IndexNow (Bing, Yandex, Seznam, Naver and others) which pages exist.
// Runs after every deploy (see .github/workflows/deploy.yml); can also be run by hand: node scripts/indexnow.mjs
// The key is public by design; the matching file lives at https://ishwarj.com/2ebfdffe3e2567fb527c5583fed2a2ad.txt
const ORIGIN = 'https://ishwarj.com'
const KEY = '2ebfdffe3e2567fb527c5583fed2a2ad'
async function get(url) {
  const response = await fetch(url, { signal: AbortSignal.timeout(30000) })
  if (!response.ok) throw new Error(`IndexNow prerequisite failed: ${url} returned ${response.status}`)
  return response.text()
}
const [xml, keyFile] = await Promise.all([get(`${ORIGIN}/sitemap.xml`), get(`${ORIGIN}/${KEY}.txt`)])
if (keyFile.trim() !== KEY) throw new Error('The live IndexNow verification file does not match the configured key.')
const urls = [...new Set([...xml.matchAll(/<loc>(.*?)<\/loc>/g)].map((m) => m[1]))]
if (!urls.length || urls.length > 10000 || urls.some((u) => new URL(u).origin !== ORIGIN)) throw new Error('Invalid or empty live sitemap; nothing was submitted.')
const res = await fetch('https://api.indexnow.org/indexnow', {
  method: 'POST', headers: { 'Content-Type': 'application/json; charset=utf-8' }, signal: AbortSignal.timeout(30000),
  body: JSON.stringify({ host: 'ishwarj.com', key: KEY, keyLocation: `${ORIGIN}/${KEY}.txt`, urlList: urls }),
})
console.log(`indexnow: submitted ${urls.length} urls, status ${res.status}`)
if (![200, 202].includes(res.status)) throw new Error(`IndexNow did not accept the submission (HTTP ${res.status}): ${(await res.text()).slice(0, 400)}`)
