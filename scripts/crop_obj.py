import os, sys

import numpy as np
import cv2
import argparse

def none_or_str(value):
    if value == 'None':
        return None
    return value

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str)
parser.add_argument('--out_name', default='images')
parser.add_argument('--mask_thres', type=float, default=127)
parser.add_argument('--bkg_color', type=none_or_str, default='0,0,0,127')
parser.add_argument('--ratio', type=float, default=1.5)

args = parser.parse_args()

froot = args.root

image_root = os.path.join(froot, 'images_raw')
mask_root = os.path.join(froot, 'masks')
out_root = os.path.join(froot, args.out_name)

ratio = args.ratio
bkg_color = [int(v) for v in args.bkg_color.split(',')] if args.bkg_color else None

os.makedirs(out_root, exist_ok=True)

img_name_list = os.listdir(image_root)
img_name_list = sorted(img_name_list)

ws = []
hs = []

for ii, fname in enumerate(img_name_list):

    if not (fname.endswith('.jpg') or fname.endswith('.png')):
        continue

    name = fname.split('.')[0]

    alpha = cv2.imread(os.path.join(mask_root, name + '.png'), -1)
    if len(alpha.shape) == 3:
        alpha = alpha[..., 0]

    yy, xx = np.where(alpha > args.mask_thres)
    y0, y1 = yy.min(), yy.max()
    x0, x1 = xx.min(), xx.max()

    ws.append(x1 - x0)
    hs.append(y1 - y0)

sz = int( max(ratio*np.max(ws), ratio*np.max(hs)) )

print('Crop size', sz)

for ii, fname in enumerate(img_name_list):

    if not (fname.endswith('.jpg') or fname.endswith('.png')):
        continue

    name = fname.split('.')[0]

    rgb = cv2.imread(os.path.join(image_root, fname))
    alpha = cv2.imread(os.path.join(mask_root, name + '.png'), -1)
    if len(alpha.shape) == 3:
        alpha = alpha[..., 0]

    rgba = np.concatenate((rgb, alpha[..., None]), axis=-1)

    yy, xx = np.where(alpha > args.mask_thres)
    y0, y1 = yy.min(), yy.max()
    x0, x1 = xx.min(), xx.max()

    height, width, chn = rgb.shape

    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2

    print(name, 'crop center', x0, y0)

    rgba_cr = rgba[y0:y1, x0:x1, :]
    out = np.zeros((sz, sz, rgba_cr.shape[-1]), dtype=rgba_cr.dtype)

    print(out.shape)

    h = rgba_cr.shape[0]
    w = rgba_cr.shape[1]

    if bkg_color is None:
        y0 = cy - int(np.floor(sz / 2))
        y1 = cy + int(np.ceil(sz / 2))
        x0 = cx - int(np.floor(sz / 2))
        x1 = cx + int(np.ceil(sz / 2))
        out = rgba[ max(y0, 0) : min(y1, height) , max(x0, 0) : min(x1, width), : ].copy()
        pads = [(max(0-y0, 0), max(y1-height, 0)), (max(0-x0, 0), max(x1-width, 0)), (0, 0)]
        out = np.pad(out, pads, mode='constant', constant_values=0)

        assert(out.shape[:2] == (sz, sz))

        out[:, :, -1] = 255
    else:
        '''black background'''
        out[ (sz - h)//2:(sz - h)//2 + h, (sz - w)//2 : (sz - w)//2 + w, : ] = rgba_cr
        out[out[:, :, -1] <= args.mask_thres] = bkg_color

    out = cv2.resize(out, (512, 512))

    cv2.imwrite(os.path.join(out_root, f'{name}.png'), out)