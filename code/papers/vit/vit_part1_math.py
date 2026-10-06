"""Part 1, the maths with real numbers.

(a) The quadratic cost of attention: how many attention weights one head of one layer needs when the tokens are
    pixels (32 x 32 CIFAR, 224 x 224 ImageNet) versus 16 x 16 patches (196 + 1 tokens), and the memory in float32.
(b) Translation equivariance of a convolution, made concrete: a 3 x 3 edge filter on a toy image and on the same image
    shifted right by 2 pixels. The output shifts by exactly 2 pixels (max difference after shifting back = 0).
(c) Patch tokens are NOT shift-equivariant: the real cat picture shifted by 1, 4, 8, 12 and 16 pixels, through the patch
    embedding layer of ViT-B/16; mean cosine similarity between the original and shifted patch vectors (after moving the
    token grid back by the whole number of patches the image moved, which is 0 for shifts under 16 and 1 for 16).
    Also the top-1 class and probability of ViT-B/16 and of ResNet-50 for every shift.
Runs on the CPU in about 30 seconds."""
import torch
import torch.nn.functional as Fn
from datasets import load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification, ResNetForImageClassification
from common import Log, save

torch.manual_seed(0)
log = Log('part1_math')
R = {}

# ---------------------------------------------------------------- (a) quadratic cost
log('(a) Attention weights per head per layer: every token looks at every token, so n tokens need n x n weights')
log(f'    {"tokens are":<34} {"n":>8} {"n x n":>16} {"float32 memory":>15}')
rows = []
for name, n in [('32 x 32 CIFAR pixels', 32 * 32), ('224 x 224 ImageNet pixels', 224 * 224),
                ('ViT-B/32 patches + [class] (7x7+1)', 7 * 7 + 1), ('ViT-B/16 patches + [class] (14x14+1)', 14 * 14 + 1),
                ('ViT-B/16 at 384 px (24x24+1)', 24 * 24 + 1)]:
    nn_ = n * n
    bytes_ = nn_ * 4
    mem = f'{bytes_ / 1e9:.2f} GB' if bytes_ >= 1e9 else (f'{bytes_ / 1e6:.2f} MB' if bytes_ >= 1e6 else f'{bytes_ / 1e3:.1f} kB')
    log(f'    {name:<34} {n:>8,} {nn_:>16,} {mem:>15}')
    rows.append(dict(name=name, n=n, nn=nn_, bytes=bytes_, mem=mem))
ratio_n = (224 * 224) / 197
ratio_cost = (224 * 224) ** 2 / 197 ** 2
log(f'    pixels / patch tokens = {224 * 224:,} / 197 = {ratio_n:.1f} x more tokens, so {ratio_cost:,.0f} x more attention weights')
log(f'    ViT-B/16 has 12 heads x 12 layers = 144 attention tables of 197 x 197 = {144 * 197 * 197:,} weights for one image')
R['cost'] = dict(rows=rows, ratio_n=round(ratio_n, 1), ratio_cost=round(ratio_cost), vit_total=144 * 197 * 197)

# ---------------------------------------------------------------- (b) translation equivariance of a convolution
log('')
log('(b) A convolution is translation equivariant: shift the input, the output shifts by the same amount')
img = torch.zeros(1, 1, 12, 12)
img[0, 0, 3:8, 2:7] = 1.0                                           # a bright 5 x 5 square at rows 3..7, columns 2..6
k = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)   # a 3 x 3 vertical-edge filter (Sobel)
shift = 2
shifted = torch.zeros_like(img)
shifted[..., shift:] = img[..., :-shift]                             # the same square, 2 pixels to the right
out = Fn.conv2d(img, k)                                              # 10 x 10 output (no padding)
out_s = Fn.conv2d(shifted, k)
diff = (out_s[..., shift:] - out[..., :-shift]).abs().max().item()   # move the second output back by 2 and compare
log(f'    toy image {tuple(img.shape[2:])}, bright square at rows 3-7, columns 2-6; filter {tuple(k.shape[2:])} (vertical edges)')
log(f'    filter weights: {k[0, 0].tolist()}')
log(f'    output {tuple(out.shape[2:])}; strongest response in the original at column {int(out[0, 0].abs().sum(0).argmax())}, in the shifted at column {int(out_s[0, 0].abs().sum(0).argmax())}')
log(f'    max |shifted output moved back by {shift} - original output| = {diff:.1f}')
log('    row 5 of the output, original: ' + ' '.join(f'{v:4.0f}' for v in out[0, 0, 5].tolist()))
log('    row 5 of the output, shifted:  ' + ' '.join(f'{v:4.0f}' for v in out_s[0, 0, 5].tolist()))
log(f'    the same filter weights ({k.numel()} numbers) are used at every one of the {out.shape[2] * out.shape[3]} output positions (weight sharing)')
R['equiv'] = dict(shift=shift, max_diff=diff, kernel=k[0, 0].tolist(), row_orig=out[0, 0, 5].tolist(), row_shift=out_s[0, 0, 5].tolist(),
                  img=img[0, 0].tolist(), out=out[0, 0].tolist(), out_shift=out_s[0, 0].tolist(), shifted_img=shifted[0, 0].tolist())

# ---------------------------------------------------------------- (c) patch tokens are not shift-equivariant
log('')
log('(c) Patch tokens are not shift equivariant: the real cat picture, shifted right by k pixels, through ViT-B/16 patch embeddings')
image = load_dataset('huggingface/cats-image', split='test')[0]['image'].convert('RGB')
proc = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').eval()
rproc = AutoImageProcessor.from_pretrained('microsoft/resnet-50')
resnet = ResNetForImageClassification.from_pretrained('microsoft/resnet-50').eval()
x = proc(images=image, return_tensors='pt').pixel_values            # (1, 3, 224, 224)
xr = rproc(images=image, return_tensors='pt').pixel_values
embed = vit.vit.embeddings.patch_embeddings                          # one 16 x 16 convolution with stride 16 = the linear patch projection
P = vit.config.patch_size


def shift_right(t, k):
    """Move the picture k pixels to the right; the strip that enters on the left repeats the edge column."""
    if k == 0:
        return t
    return torch.cat([t[..., :1].expand(*t.shape[:-1], k), t[..., :-k]], dim=-1)


def top1(model, t):
    with torch.no_grad():
        p = torch.softmax(model(pixel_values=t).logits[0], -1)
    v, i = p.max(0)
    return model.config.id2label[int(i)].split(',')[0], round(float(v), 4)


with torch.no_grad():
    e0 = embed(x)[0]                                                 # (196, 768): one vector per patch
grid = e0.view(14, 14, -1)
log(f'    patch embedding layer: {embed.projection}')
log(f'    patch vectors: {tuple(e0.shape)} = 196 patches x 768 numbers, arranged on a 14 x 14 grid')
log(f'    {"shift k":>8} {"grid moved back":>16} {"mean cosine":>12} {"min cosine":>11}   {"ViT-B/16 top-1":<26} {"ResNet-50 top-1":<26}')
rows = []
for k in [0, 1, 4, 8, 12, 16]:
    with torch.no_grad():
        ek = embed(shift_right(x, k))[0].view(14, 14, -1)
    back = k // P                                                    # whole patches the picture moved
    a, b = grid[:, :14 - back], ek[:, back:]                        # compare patch (i, j) of the original with patch (i, j + back) of the shifted
    cos = Fn.cosine_similarity(a.reshape(-1, 768), b.reshape(-1, 768), dim=-1)
    tv, tr = top1(vit, shift_right(x, k)), top1(resnet, shift_right(xr, k))
    log(f'    {k:>8} {back:>16} {cos.mean():>12.3f} {cos.min():>11.3f}   {tv[0] + " " + f"{tv[1]:.3f}":<26} {tr[0] + " " + f"{tr[1]:.3f}":<26}')
    rows.append(dict(k=k, back=back, mean_cos=round(float(cos.mean()), 4), min_cos=round(float(cos.min()), 4), vit=tv, resnet=tr))
same16 = Fn.cosine_similarity(grid.reshape(-1, 768), embed(shift_right(x, 16))[0].detach(), dim=-1).mean()
log(f'    for k = 16 compared at the SAME grid position (no moving back): mean cosine = {same16:.3f}  (each slot now holds its left neighbour)')
log('    a 16-pixel shift moves whole patches, so the token vectors are the same, just one slot over; smaller shifts change every vector')
R['shift'] = dict(rows=rows, same_pos_16=round(float(same16), 4), patch=P)
save('part1_math', R)
