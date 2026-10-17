# Search discovery and indexing

The canonical site is https://ishwarj.com/. HTTP and `www` redirect to it. Public page URLs end in `/`; source filenames and existing section anchors remain stable.

## What the build provides

- Complete HTML text, navigation and internal links without JavaScript on every published route.
- Unique page titles and descriptions; canonical URLs; Open Graph and X cards; the same complete metadata after client navigation.
- `Person`, `WebSite`, `WebPage`, article, book, video and breadcrumb structured data. Original paper authors are separate from the author of the explanations. A YouTube watch URL is not advertised as a downloadable video file.
- Visible author, source, update date and a canonical citation on article pages.
- `sitemap.xml` with source-derived dates, a full-content-type RSS feed, and a real `404.html` marked `noindex` without a home-page canonical.
- `llms.txt` and a Markdown alternative for each published content page. These are convenience formats for readers and compatible tools, not ranking directives or guarantees that a service will read them.
- Public crawling, including `OAI-SearchBot`. The existing general crawler policy remains open. Search access and model-training crawler policies are separate decisions; this change does not add new training-specific grants or restrictions.
- An automated check of every canonical page, metadata, structured data, internal link, section anchor, Markdown alternative and client head replacement. Deployment runs this check before uploading the site.

## One-time owner setup

1. Add `ishwarj.com` as a domain property in [Google Search Console](https://search.google.com/search-console). Verify using the DNS record Google provides. Alternatively, add the HTTPS URL-prefix property and use HTML-tag verification: put the exact tag's `content` value in the GitHub Actions repository variable `GOOGLE_SITE_VERIFICATION`, redeploy, then verify.
2. Submit `https://ishwarj.com/sitemap.xml` in Search Console. Inspect the homepage, a book chapter, an article and a paper explanation. Request indexing for a few representative pages, then monitor coverage and excluded-page reasons.
3. Add the site to [Bing Webmaster Tools](https://www.bing.com/webmasters/). Import the verified Search Console property or verify directly. For HTML-tag verification, put the exact `msvalidate.01` content value in the repository variable `BING_SITE_VERIFICATION`, redeploy and verify. Submit the same sitemap.
4. Review Bing's IndexNow report and Google's indexing report after future deployments. A successful submission means the URLs were accepted for consideration, not that they are indexed.

The public IndexNow verification key already lives in `public/`. The deployment validates the live sitemap and key, submits the canonical URLs, and reports rejection or timeout as a workflow warning. It does not submit to Google, which uses the sitemap and its own crawling.

## Publishing conventions

- Write a specific `title` and `description` in frontmatter when the automatic first-paragraph summary is not useful. Keep the actual topic in the title; there is no forced title truncation.
- Use `date` for the original publication date and `updated` for a known substantive update. Git source history supplies missing modification dates; it does not invent an original publication date. Never update dates solely to appear fresh. Keep full history in CI; builds without history omit unknown dates.
- Link claims to sources and explain the conditions behind measurements. Keep runnable code and original-paper attribution alongside explanations. Do not fabricate FAQs, reviews, ratings, authorship or results for structured data.
- Use relative `.md` links inside the repo; the builder resolves published targets to website URLs. Preserve published filenames and heading text when possible; changing them changes URLs or anchors.
- Production builds exclude `draft: true` content and hidden books/papers, including from HTML, feeds, the sitemap and Markdown exports. Do not use a public repo as storage for confidential drafts.
- Put new social-preview images in `public/og/`; the site card is the fallback. Run `npm run build && npm run check:seo` before publishing.
- Link to the website's canonical page when promoting an article. Use section anchors for precise references. The repository is the source/code companion.

## Validate a release

```sh
npm ci
npm run typecheck
NODE_ENV=production npm run build
npm run check:seo
```

After deployment, verify that `robots.txt`, `sitemap.xml`, `rss.xml`, `llms.txt` and representative articles return HTTP 200, and an unknown URL returns HTTP 404. Check representative pages in Google's Rich Results Test and Search Console's URL Inspection. Watch crawl/indexing reports and real referrals rather than treating a crawler allowance or text index as proof of visibility.

References: [Google AI features](https://developers.google.com/search/docs/appearance/ai-features), [Google sitemaps](https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap), [OpenAI crawlers](https://developers.openai.com/api/docs/bots), [Bing URL submission](https://www.bing.com/webmasters/help/URL-Submission-62f2860b).
