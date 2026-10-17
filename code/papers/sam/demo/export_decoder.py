"""Export the prompt encoder + mask decoder of the released SAM ViT-H (facebook/sam-vit-huge) to one ONNX graph
for the browser demo at /papers/sam/demo.

Inputs of the graph (the same convention as the ONNX export of the official segment-anything repository):
  image_embeddings  (1, 256, 64, 64)  float32   the output of the image encoder (precomputed by embed_images.py)
  point_coords      (1, N, 2)         float32   points in the 1024 frame (photo pixel x 1024 / longest side)
  point_labels      (1, N)            float32   1 foreground, 0 background, 2 box top-left, 3 box bottom-right,
                                                -1 padding ("not a point", added when there is no box)
  mask_input        (1, 1, 256, 256)  float32   low-res logits of a previous mask (the mask prompt)
  has_mask_input    (1,)              float32   1 to use mask_input, 0 for the learned "no mask" vector
Outputs:
  low_res_logits    (1, 4, 256, 256)  the four mask tokens: 0 = the single-mask output, 1..3 = the three masks
  iou_predictions   (1, 4)            the predicted IoU of each of the four masks

The graph re-uses the weights of transformers' SamModel and follows its code line by line (SamPromptEncoder,
SamMaskDecoder); only the final "keep masks 1..3" slice is left out, so the browser gets all four tokens.
The check at the end compares this wrapper with SamModel itself in PyTorch, then the .onnx file (onnxruntime, if
installed: pip install onnx onnxruntime) with the wrapper. reference.py and check_browser.mjs check the browser.

Writes public/demos/sam/sam_decoder.onnx and code/papers/sam/demo/results/export_stdout.txt."""
import os, sys, json
import numpy as np
import torch
from torch import nn
from transformers import SamModel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', '..', '..', 'public', 'demos', 'sam')
RES = os.path.join(HERE, 'results')
os.makedirs(OUT, exist_ok=True)
os.makedirs(RES, exist_ok=True)


class PromptAndDecoder(nn.Module):
    def __init__(self, sam: SamModel):
        super().__init__()
        self.pe = sam.prompt_encoder
        self.dec = sam.mask_decoder
        self.shared = sam.shared_image_embedding        # the random matrix G of the positional encoding (2 x 128)
        # the positional encoding of the 64 x 64 grid is a fixed function of G. The graph computes it from the
        # cell centres (8,192 numbers) instead of storing its 1,048,576 numbers: a 4 MB smaller file.
        g = torch.ones(64, 64).cumsum(0) - 0.5
        self.register_buffer('grid_xy', torch.stack([g.T, g], dim=-1) / 64)   # (64, 64, 2): (x, y) of each cell centre

    def embed_points(self, coords, labels):
        coords = coords + 0.5                                              # centre of the pixel
        coords = coords / 1024.0
        pe = self.shared(coords[:, None])[:, 0]                            # (1, N, 256): sin and cos of 2 pi (2u - 1) G
        lab = labels[..., None]
        emb = torch.where(lab == -1, self.pe.not_a_point_embed.weight.expand_as(pe), pe)
        for k in range(4):                                                 # add the learned type vector of each label
            emb = emb + (lab == k).float() * self.pe.point_embed[k].weight
        return emb

    def forward(self, image_embeddings, point_coords, point_labels, mask_input, has_mask_input):
        sparse = self.embed_points(point_coords, point_labels)[:, None]   # (1, 1, N, 256)
        dense = self.pe.mask_embed(mask_input) * has_mask_input.reshape(1, 1, 1, 1) + \
            (1 - has_mask_input.reshape(1, 1, 1, 1)) * self.pe.no_mask_embed.weight.reshape(1, -1, 1, 1)
        d = self.dec
        b, c, h, w = image_embeddings.shape
        tokens = torch.cat([d.iou_token.weight, d.mask_tokens.weight], dim=0)[None, None]   # (1, 1, 5, 256)
        tokens = torch.cat([tokens, sparse], dim=2)
        img = image_embeddings + dense
        image_pe = self.shared(self.grid_xy).permute(2, 0, 1)[None]       # (1, 256, 64, 64)
        point_embedding, img = d.transformer(point_embeddings=tokens, image_embeddings=img,
                                             image_positional_embeddings=image_pe,
                                             attention_similarity=None)
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1:5, :]
        img = img.transpose(2, 3).reshape(b, c, h, w)
        up = d.upscale_conv1(img)
        up = d.activation(d.upscale_layer_norm(up))
        up = d.activation(d.upscale_conv2(up))
        hyper = torch.stack([d.output_hypernetworks_mlps[i](mask_tokens_out[:, :, i, :]) for i in range(4)], dim=2)
        _, cc, hh, ww = up.shape
        masks = (hyper @ up.reshape(b, 1, cc, hh * ww)).reshape(b, 4, hh, ww)
        iou = d.iou_prediction_head(iou_token_out).reshape(b, 4)
        return masks, iou


def main():
    torch.manual_seed(0)
    sam = SamModel.from_pretrained('facebook/sam-vit-huge', attn_implementation='eager').eval().requires_grad_(False)
    model = PromptAndDecoder(sam).eval()

    # ---- check the wrapper against SamModel in PyTorch, on a real image embedding (the cats photo)
    emb_path = os.path.expanduser('~/.cache/papers/sam_demo/emb_cats_fp32.npy')
    if not os.path.exists(emb_path):
        sys.exit('run embed_images.py first (it saves the fp32 embedding of the cats photo)')
    emb = torch.from_numpy(np.load(emb_path))
    s = 1024 / 640
    cases = {
        'ear click': dict(pts=[[275, 62]], lab=[1]),
        'ear + body, background on the sofa': dict(pts=[[275, 62], [120, 220], [500, 420]], lab=[1, 1, 0]),
        'box around the right cat': dict(box=[255, 20, 640, 360]),
        'box + point': dict(pts=[[400, 150]], lab=[1], box=[255, 20, 640, 360]),
    }
    log = []
    worst = 0.0
    for name, cse in cases.items():
        kw = {}
        coords, labels = [], []
        if 'pts' in cse:
            kw['input_points'] = torch.tensor([[[[x * s, y * s] for x, y in cse['pts']]]], dtype=torch.float32)
            kw['input_labels'] = torch.tensor([[cse['lab']]], dtype=torch.int64)
            coords += [[x * s, y * s] for x, y in cse['pts']]
            labels += cse['lab']
        if 'box' in cse:
            x0, y0, x1, y1 = cse['box']
            kw['input_boxes'] = torch.tensor([[[x0 * s, y0 * s, x1 * s, y1 * s]]], dtype=torch.float32)
            coords += [[x0 * s, y0 * s], [x1 * s, y1 * s]]
            labels += [2, 3]
        else:
            coords += [[0.0, 0.0]]
            labels += [-1]
        ref3 = sam(image_embeddings=emb, multimask_output=True, **kw)
        ref1 = sam(image_embeddings=emb, multimask_output=False, **kw)
        m, iou = model(emb, torch.tensor([coords], dtype=torch.float32), torch.tensor([labels], dtype=torch.float32),
                       torch.zeros(1, 1, 256, 256), torch.zeros(1))
        dm = max((m[:, 1:] - ref3.pred_masks[0, 0]).abs().max().item(), (m[:, :1] - ref1.pred_masks[0, 0]).abs().max().item())
        di = max((iou[:, 1:] - ref3.iou_scores[0, 0]).abs().max().item(), (iou[:, :1] - ref1.iou_scores[0, 0]).abs().max().item())
        worst = max(worst, dm)
        log.append(f'{name:<38} logits max diff {dm:.2e}   scores max diff {di:.2e}')
    # mask prompt: feed back the ear's best mask
    kw = dict(input_points=torch.tensor([[[[275 * s, 62 * s]]]]), input_labels=torch.tensor([[[1]]]))
    first = sam(image_embeddings=emb, multimask_output=True, **kw)
    prev = first.pred_masks[0, 0, first.iou_scores[0, 0].argmax()][None, None]
    ref = sam(image_embeddings=emb, multimask_output=True, input_masks=prev, **kw)
    m, iou = model(emb, torch.tensor([[[275 * s, 62 * s], [0.0, 0.0]]]), torch.tensor([[1.0, -1.0]]), prev, torch.ones(1))
    dm = (m[:, 1:] - ref.pred_masks[0, 0]).abs().max().item()
    log.append(f'{"ear click + previous mask as prompt":<38} logits max diff {dm:.2e}   scores max diff '
               f'{(iou[:, 1:] - ref.iou_scores[0, 0]).abs().max().item():.2e}')
    for line in log:
        print(line)

    # ---- export
    path = os.path.join(OUT, 'sam_decoder.onnx')
    args = (emb, torch.tensor([[[440.0, 99.2], [0.0, 0.0]]]), torch.tensor([[1.0, -1.0]]),
            torch.zeros(1, 1, 256, 256), torch.zeros(1))
    torch.onnx.export(model, args, path, dynamo=False, opset_version=17,
                      input_names=['image_embeddings', 'point_coords', 'point_labels', 'mask_input', 'has_mask_input'],
                      output_names=['low_res_logits', 'iou_predictions'],
                      dynamic_axes={'point_coords': {1: 'n'}, 'point_labels': {1: 'n'}},
                      do_constant_folding=False)   # keep the grid encoding as a computation
    print(f'saved {path}: {os.path.getsize(path) / 1e6:.2f} MB')

    # the .onnx file against the PyTorch wrapper, with onnxruntime (CPU) on the same inputs
    try:
        import onnxruntime as ortp
        sess = ortp.InferenceSession(path, providers=['CPUExecutionProvider'])
        for name, (c, l, mi, hm) in {
            'ear click': ([[440.0, 99.2], [0.0, 0.0]], [1.0, -1.0], torch.zeros(1, 1, 256, 256), 0.0),
            'ear + body + background': ([[440.0, 99.2], [192.0, 352.0], [800.0, 672.0], [0.0, 0.0]],
                                        [1.0, 1.0, 0.0, -1.0], torch.zeros(1, 1, 256, 256), 0.0),
            'box': ([[408.0, 32.0], [1024.0, 576.0]], [2.0, 3.0], torch.zeros(1, 1, 256, 256), 0.0),
            'ear click + previous mask': ([[440.0, 99.2], [0.0, 0.0]], [1.0, -1.0], prev, 1.0),
        }.items():
            m, iou = model(emb, torch.tensor([c]), torch.tensor([l]), mi, torch.tensor([hm]))
            om, oi = sess.run(None, dict(image_embeddings=emb.numpy(), point_coords=np.array([c], np.float32),
                                         point_labels=np.array([l], np.float32), mask_input=mi.numpy(),
                                         has_mask_input=np.array([hm], np.float32)))
            agree = ((om > 0) == (m.numpy() > 0)).mean()
            line = (f'onnxruntime vs PyTorch, {name:<26} logits max diff {np.abs(om - m.numpy()).max():.2e}  '
                    f'scores max diff {np.abs(oi - iou.numpy()).max():.2e}  same sign on {100 * agree:.4f}% of pixels')
            print(line)
            log.append(line)
    except ImportError:
        log.append('onnxruntime not installed: the .onnx file is checked in the browser only (check_browser.mjs)')
    with open(os.path.join(RES, 'export_stdout.txt'), 'w') as f:
        f.write('wrapper vs transformers SamModel (PyTorch, cats photo):\n' + '\n'.join(log) + '\n')
        f.write(f'saved sam_decoder.onnx: {os.path.getsize(path):,} bytes\n')


if __name__ == '__main__':
    with torch.no_grad():
        main()
