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
}
