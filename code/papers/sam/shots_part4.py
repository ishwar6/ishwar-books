"""Highlighted excerpts of the SAM paper for Part 4 (the data engine and SA-1B): Section 4, Section 5, Section 6,
Figures 2, 5, 6 and 7, Tables 1, 2 and 6, Appendix B and Appendix C."""
import os
from paper_shots import run, OUT

JOBS = [
    # ---------------------------------------------------------------- page 3: Figure 2 (the bottom rows)
    dict(name='p4-figure2', page=2, anchor='Figure 2: Example images with overlaid masks from our newly introduced dataset',
         highlight=['annotated fully automatically by SAM', '∼100 masks per image on average'], figure=True, column='full', below=0, fig_top=250),
    # ---------------------------------------------------------------- page 5: Section 4, the first stage
    dict(name='p4-engine-intro', page=4, anchor='As segmentation masks are not abundant on the inter',
         highlight=['segmentation masks are not abundant', '1.1B mask dataset, SA-1B', 'model-assisted manual annotation stage',
                    'semi-automatic stage', 'fully automatic stage'],
         above=20, end='details of each next.'),
    dict(name='p4-assisted-1', page=4, anchor='Assisted-manual stage. In the first stage',
         highlight=['professional annotators', 'clicking foreground / background ob', 'pixel-precise “brush” and “eraser” tools',
                    'did not impose semantic constraints', '“stuff” and “things”', 'did not collect these names',
                    'order of prominence', 'over 30 seconds'],
         above=4, end='took over 30 seconds to annotate.'),
    # ---------------------------------------------------------------- page 6
    dict(name='p4-assisted-2', page=5, anchor='At the start of this stage, SAM was trained using',
         highlight=['common public segmentation datasets', 'retrained using only newly annotated', 'scaled from ViT-B to ViT-H',
                    'retrained our model 6 times', 'decreased from 34 to 14', '6.5× faster than mask annotation for COCO',
                    'only 2× slower than bounding-box labeling', 'increased from 20 to 44 masks', '4.3M masks from 120k images'],
         above=4, end='120k images in this stage.'),
    dict(name='p4-semi', page=5, anchor='Semi-automatic stage. In this stage, we aimed',
         highlight=['increase the diversity of masks', 'less prominent objects', 'automatically detected conﬁdent', 'preﬁlled',
                    'bounding box detector [84]', 'generic “object” category', '5.9M masks in 180k images', '10.2M',
                    '(5 times)', 'went back up to 34 seconds', 'from 44 to 72 masks'],
         above=4, end='(including the automatic masks).'),
    dict(name='p4-auto-1', page=5, anchor='Fully automatic stage. In the ﬁnal stage',
         highlight=['annotation was fully automatic', 'collected enough masks', 'ambiguity-aware', '32×32 regular grid of points',
                    'return the subpart, part, and whole ob'],
         above=4, end='part, and whole object.'),
    dict(name='p4-auto-2', page=5, anchor='The IoU prediction module of our model is used to se',
         highlight=['select conﬁdent masks', 'only stable masks', 'thresholding the probability map at 0.5 −δ and 0.5 + δ',
                    'non-maximal suppression (NMS)', 'ﬁlter duplicates', 'zoomed-in', 'producing a total of 1.1B high-quality masks'],
         above=4, end='We describe and analyze the resulting dataset, SA-1B, next.'),
    dict(name='p4-figure5', page=5, anchor='Figure 5: Image-size normalized mask center distributions',
         highlight=['Image-size normalized mask center distributions'], figure=True, below=0),
    # ---------------------------------------------------------------- page 6: Section 5
    dict(name='p4-dataset', page=5, anchor='Our dataset, SA-1B, consists of 11M diverse',
         highlight=['11M diverse, high-', 'licensed, and privacy protecting', '1.1B high-quality segmentation masks'],
         above=20, end='with protections for researchers.'),
    dict(name='p4-images', page=5, anchor='Images. We licensed a new set of 11M images',
         highlight=['3300×4950 pixels on average', 'shortest side set to 1500 pixels', '∼480×640 pixels',
                    'Faces and vehicle', 'license plates have been blurred'],
         above=4, end='blurred in the released images.'),
    dict(name='p4-masks', page=5, anchor='Masks. Our data engine produced 1.1B masks',
         highlight=['99.1%', 'generated fully automatically', 'centrally important', 'SA-1B only includes',
                    'automatically generated masks'],
         above=4, end='automatically generated masks.'),
    dict(name='p4-quality', page=5, anchor='Mask quality. To estimate mask quality',
         highlight=['500 images (∼50k masks)', 'improve the quality of all masks', 'pairs of automatically predicted and professionally cor',
                    '94% of pairs have greater than 90% IoU', '97% of pairs have greater than 75% IoU', 'inter-annotator consistency at 85-91%'],
         above=4, end='produced by the data engine.'),
    # ---------------------------------------------------------------- page 7
    dict(name='p4-figure6', page=6, anchor='Figure 6: Dataset mask properties',
         highlight=['11× more images and 400× more masks'], figure=True, column='full', below=0),
    dict(name='p4-properties', page=6, anchor='Mask properties. In Fig. 5 we plot the spatial',
         highlight=['photographer biases', 'greater', 'coverage of image corners', 'prominent center bias', '36× more masks per image',
                    '3.5× fewer masks per image', 'square root of the mask area divided by image area',
                    '1 minus mask area divided by area of mask’s convex hull', 'stratiﬁed sampling', 'broadly similar'],
         hl_nth={'greater': 0},
         above=4, end='similar to that of other datasets.'),
    dict(name='p4-figure7', page=6, anchor='Figure 7: Estimated geographic distribution of SA-1B images',
         highlight=['more than 1000 images', 'three countries with the most images are from different parts of the world'],
         figure=True, column='full', below=0, fig_top=86),
    dict(name='p4-rai', page=6, anchor='We next perform a Responsible AI (RAI) analysis',
         highlight=['potential fairness concerns and bi', 'geographic and income distribution of SA-1B', 'fairness of',
                    'protected attributes of people', 'model cards in §F'],
         above=22, end='model cards in §F.'),
    dict(name='p4-table1', page=6, anchor='Table 1: Comparison of geographic and income representa',
         highlight=['higher representation in Europe and Asia &', 'underrepresented in all datasets'], figure=True, below=0, fig_top=126),
    dict(name='p4-geo', page=6, anchor='Geographic and income representation. We infer the',
         highlight=['infer the', 'top-three countries are', 'substan', 'underrepresent Africa as well as low income countries',
                    'at least 28 million masks', '94-108 per image'],
         hl_nth={'infer the': 0},
         above=4, end='(94-108 per image).'),
    # ---------------------------------------------------------------- page 8
    dict(name='p4-table2', page=7, anchor='Table 2: SAM’s performance segmenting people across',
         highlight=['95% conﬁdence intervals', 'all conﬁdence intervals overlap except older vs. middle'], figure=True, below=0, fig_top=110),
    dict(name='p4-fairness', page=7, anchor='Fairness in segmenting people. We investigate potential',
         highlight=['across perceived gender presentation', 'perceived age group', 'perceived skin tone', 'performance discrepancy',
                    'random sampling of 1', 'SAM performs similarly across groups', 'Fitzpatrick skin type [36]',
                    'do not ﬁnd a signiﬁcant difference', 'biases may arise when SAM is', 'indication of bias across perceived gender presentation'],
         hl_nth={'across perceived gender presentation': 1, 'perceived age group': 1, 'perceived skin tone': 1},
         above=4, end='bias across perceived gender presentation.'),
    # ---------------------------------------------------------------- page 18: Appendix B and C
    dict(name='p4-b-cropping', page=17, anchor='Cropping. Masks were generated from a regular grid',
         highlight=['32×32 points on the full image', '20 additional zoomed-', '2×2 and 4×4 partially over', '16×16 and 8×8 regular point grids',
                    'removed masks that touch the inner boundaries', 'greedy box-based NMS', 'predicted IoU to rank masks',
                    'most zoomed-in', 'NMS threshold of 0.7'],
         above=4, end='NMS threshold of 0.7.'),
    dict(name='p4-b-filtering', page=17, anchor='Filtering. We used three ﬁlters to increase mask',
         highlight=['predicted IoU score at a threshold of 88.0', 'thresholding logits at 0', 'pair of -1 and +1 thresholded masks',
                    'equal to or greater than 95.0', 'covered 95% or more of an', 'judged by'],
         above=4, end='described in §5.'),
    dict(name='p4-b-post-a', page=17, anchor='Postprocessing. We observed two error types',
         highlight=['estimated 4% of masks include small, spurious components', 'removed connected components with area less'],
         above=4, end='connected components with area less'),
    dict(name='p4-b-post-b', page=17, anchor='than 100 pixels (including removing entire masks',
         highlight=['than 100 pixels', 'small, spurious holes', 'ﬁlled holes with area less than 100 pixels'],
         hl_nth={'than 100 pixels': 0}, above=4, end='components of inverted masks.'),
    dict(name='p4-b-model', page=17, anchor='Automatic mask generation model. We trained a special',
         highlight=['special', 'trained on manual and semi-automatic data only', '177656 iterations instead of 90k',
                    'only point and mask prompts (no boxes)', 'mask decoder used 3 layers instead of 2'],
         hl_nth={'special': 0},
         above=4, end='used 3 layers instead of 2.'),
    dict(name='p4-c-geo', page=17, anchor='Inferring geographic information for SA-1B',
         highlight=['not geo-tagged', 'caption describing its contents and where it was taken', 'named entity recognition model',
                    '“Geor-', 'do not release the inferred locations'],
         above=4, end='required by the image provider.'),
    dict(name='p4-table6', page=18, anchor='Table 6: SAM’s performance segmenting clothing across',
         highlight=['The intervals', 'for perceived gender are disjoint', 'masculine being higher'], figure=True, below=0, fig_top=71),
    dict(name='p4-c-clothing', page=18, anchor='Fairness in segmenting clothing. We extend our analysis',
         highlight=['6.5k ground truth masks', 'better at segment', 'disjoint 95% conﬁdence intervals', 'The gap closes',
                    'there is a bias when segmenting clothing'],
         above=4, end='mindful of this limitation.'),
]

def stitch(parts, name):
    """Put excerpts from two columns one under the other, as one picture."""
    from PIL import Image
    ims = [Image.open(os.path.join(OUT, p + '.png')) for p in parts]
    out = Image.new('RGB', (max(i.width for i in ims), sum(i.height for i in ims)), (255, 255, 255))
    y = 0
    for i in ims:
        out.paste(i, (0, y)); y += i.height
    out.save(os.path.join(OUT, name + '.png'))
    for p in parts:
        os.remove(os.path.join(OUT, p + '.png'))
    print(f'{name}: {out.width}x{out.height} from {", ".join(parts)}')


if __name__ == '__main__':
    run(JOBS)
    if not os.environ.get('ONLY') or 'p4-b-post' in os.environ.get('ONLY'):
        stitch(['p4-b-post-a', 'p4-b-post-b'], 'p4-b-post')
