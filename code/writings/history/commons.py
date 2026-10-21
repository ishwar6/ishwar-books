"""Look up Wikimedia Commons files: search, then print size, licence and author for each hit."""
import json, sys, time, urllib.parse, urllib.request
UA = {'User-Agent': 'ishwarj-history-article/1.0 (iisudrj11@gmail.com)'}
def api(**kw):
    kw['format'] = 'json'
    url = 'https://commons.wikimedia.org/w/api.php?' + urllib.parse.urlencode(kw)
    for i in range(4):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA)))
        except Exception as e:
            time.sleep(3 + 3 * i)
    raise SystemExit('failed ' + url)
def info(titles):
    d = api(action='query', titles='|'.join(titles), prop='imageinfo', iiprop='url|size|extmetadata')
    for p in d['query']['pages'].values():
        if 'imageinfo' not in p: print(p['title'], 'MISSING'); continue
        i = p['imageinfo'][0]; m = i['extmetadata']
        g = lambda k: (m.get(k, {}).get('value', '') or '').replace('\n', ' ')[:110]
        print(f"{p['title']} | {i['width']}x{i['height']} | {g('LicenseShortName')} | {g('Artist')} | {i['url']}")
if __name__ == '__main__':
    if sys.argv[1] == 'search':
        for q in sys.argv[2:]:
            d = api(action='query', list='search', srnamespace=6, srlimit=8, srsearch=q)
            t = [x['title'] for x in d['query']['search']]
            print('==', q); info(t) if t else None; time.sleep(1.5)
    else:
        info(sys.argv[2:])
