// Tell search engines that use IndexNow (Bing, Yandex, Seznam, Naver and others) which pages exist.
// Runs after every deploy (see .github/workflows/deploy.yml); can also be run by hand: node scripts/indexnow.mjs
// The key is public by design; the matching file lives at https://ishwarj.com/2ebfdffe3e2567fb527c5583fed2a2ad.txt
const ORIGIN = 'https://ishwarj.com'
const KEY = '2ebfdffe3e2567fb527c5583fed2a2ad'
const xml = await (await fetch(`${ORIGIN}/sitemap.xml`)).text()
const urls = [...xml.matchAll(/<loc>(.*?)<\/loc>/g)].map((m) => m[1])
const res = await fetch('https://api.indexnow.org/indexnow', {
  method: 'POST', headers: { 'Content-Type': 'application/json; charset=utf-8' },
  body: JSON.stringify({ host: 'ishwarj.com', key: KEY, keyLocation: `${ORIGIN}/${KEY}.txt`, urlList: urls }),
})
console.log(`indexnow: submitted ${urls.length} urls, status ${res.status}`)
