"""Part 1: one real picture through ViT-B/16 and through ResNet-50.

Loads google/vit-base-patch16-224 (pre-trained on ImageNet-21k, fine-tuned on ImageNet-1k) and one picture
(the huggingface/cats-image photo), saves the picture for the page, and prints:
  the image tensor shape (3 x 224 x 224 = 150,528 numbers), the number of 16 x 16 patches (196), the numbers per
  patch (16 x 16 x 3 = 768), the sequence length with the [class] token (197), and the top-5 ImageNet classes.
Then runs microsoft/resnet-50 (a convolutional network) on the same picture, so a CNN and a Transformer can be compared.
Runs on the CPU in about 20 seconds; the models are about 350 MB (ViT) and 100 MB (ResNet) the first time."""
import os
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification, ResNetForImageClassification
from common import Log, save

torch.manual_seed(0)
log = Log('part1')
IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'public', 'img', 'papers', 'vit')
R = {}

# ---------------------------------------------------------------- the picture
image = load_dataset('huggingface/cats-image', split='test')[0]['image'].convert('RGB')
log(f'picture: {image.size[0]} x {image.size[1]} pixels, mode {image.mode} (3 colour channels: red, green, blue)')
small = image.copy()
small.thumbnail((480, 480))
os.makedirs(IMG_DIR, exist_ok=True)
small.save(os.path.join(IMG_DIR, 'p1-sample.png'), optimize=True)
R['picture'] = dict(width=image.size[0], height=image.size[1], saved=f'{small.size[0]}x{small.size[1]}')

# ---------------------------------------------------------------- ViT-B/16
proc = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').eval()
x = proc(images=image, return_tensors='pt').pixel_values            # resized to 224 x 224, scaled to [-1, 1]
C, H, W = x.shape[1:]
P = vit.config.patch_size
N = (H // P) * (W // P)
log('')
log('ViT-B/16 (google/vit-base-patch16-224)')
log(f'  image tensor: {tuple(x.shape)}  = batch of 1, {C} channels, {H} x {W} pixels')
log(f'  numbers in one image: {C} x {H} x {W} = {C * H * W:,}')
log(f'  patch size P = {P}; patches per side = {H} // {P} = {H // P}; number of patches N = {H // P} x {W // P} = {N}')
log(f'  numbers in one patch: {P} x {P} x {C} = {P * P * C}')
log(f'  sequence length with the [class] token: {N} + 1 = {N + 1}')
log(f'  hidden size D = {vit.config.hidden_size}, layers = {vit.config.num_hidden_layers}, heads = {vit.config.num_attention_heads}')
log(f'  parameters: {sum(p.numel() for p in vit.parameters()):,}')
with torch.no_grad():
    out = vit(pixel_values=x, output_hidden_states=True)
log(f'  token matrix after the embedding step: {tuple(out.hidden_states[0].shape)}  (1 image, {N + 1} tokens, {vit.config.hidden_size} numbers each)')
log(f'  output scores: {tuple(out.logits.shape)}  (one score per ImageNet class)')
p_vit = torch.softmax(out.logits[0], -1)
vals, idx = p_vit.topk(5)
top_vit = [(vit.config.id2label[int(i)].split(',')[0], round(float(v), 4)) for v, i in zip(vals, idx)]
log('  top-5 classes:')
for name, p in top_vit:
    log(f'    {p:7.4f}  {name}')
R['vit'] = dict(shape=list(x.shape), numbers=C * H * W, patch=P, n_patches=N, per_patch=P * P * C, seq_len=N + 1,
                hidden=vit.config.hidden_size, layers=vit.config.num_hidden_layers, heads=vit.config.num_attention_heads,
                params=sum(p.numel() for p in vit.parameters()), top5=top_vit)

# ---------------------------------------------------------------- ResNet-50
rproc = AutoImageProcessor.from_pretrained('microsoft/resnet-50')
resnet = ResNetForImageClassification.from_pretrained('microsoft/resnet-50').eval()
xr = rproc(images=image, return_tensors='pt').pixel_values
log('')
log('ResNet-50 (microsoft/resnet-50), a convolutional network')
log(f'  image tensor: {tuple(xr.shape)}')
log(f'  parameters: {sum(p.numel() for p in resnet.parameters()):,}')
with torch.no_grad():
    ro = resnet(pixel_values=xr, output_hidden_states=True)
for i, h in enumerate(ro.hidden_states):
    log(f'  feature map after stage {i}: {tuple(h.shape[1:])}  (channels, height, width)')
p_res = torch.softmax(ro.logits[0], -1)
vals, idx = p_res.topk(5)
top_res = [(resnet.config.id2label[int(i)].split(',')[0], round(float(v), 4)) for v, i in zip(vals, idx)]
log('  top-5 classes:')
for name, p in top_res:
    log(f'    {p:7.4f}  {name}')
R['resnet'] = dict(shape=list(xr.shape), params=sum(p.numel() for p in resnet.parameters()),
                   stages=[list(h.shape[1:]) for h in ro.hidden_states], top5=top_res)

log('')
log(f'both models agree on the top class: {top_vit[0][0] == top_res[0][0]}  (ViT: {top_vit[0][0]}, ResNet-50: {top_res[0][0]})')
R['agree'] = top_vit[0][0] == top_res[0][0]
save('part1', R)
