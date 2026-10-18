type DemoGuide = { category: string; title: string; summary: string; mode: string; actions: string[] }

export const demoGuides: Record<string, DemoGuide> = {
  sam: {
    category: 'Image segmentation',
    title: 'Click a point. Find the object.',
    summary: 'Tell Segment Anything what to look for. Refine its masks with a click, a box, or a background point.',
    mode: 'Live model · sample photos',
    actions: ['Select an object', 'Compare three masks', 'Segment the whole image'],
  },
  vit: {
    category: 'Image classification',
    title: 'See a picture become a prediction.',
    summary: 'Follow an image through a Vision Transformer, from tiny patches to attention maps and its final answer.',
    mode: 'Saved runs + live model',
    actions: ['Inspect image patches', 'Explore attention', 'Try your own photo'],
  },
  transformer: {
    category: 'Translation',
    title: 'Watch a Transformer write, one token at a time.',
    summary: 'Type a date and follow it through a small Transformer in your browser, look inside a real English to German translator, and work attention by hand.',
    mode: 'Live model + saved runs',
    actions: ['Step through the decoder', 'Explore attention heads', 'Break the positions or the mask'],
  },
  'meta-harness': {
    category: 'Harness search',
    title: 'Open the folder an agent searches through.',
    summary: 'Browse a real harness search: every candidate\'s code, traces and score. Choose what the proposer may read, and explore the paper\'s results.',
    mode: 'Saved search run + paper numbers',
    actions: ['Browse candidates and diffs', 'See the overfitting gap', 'Scores vs summaries vs traces'],
  },
}
