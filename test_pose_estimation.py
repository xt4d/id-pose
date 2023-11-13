import os, sys
import argparse
import json

import torch
import numpy as np
from omegaconf import OmegaConf
import torchvision.utils as vutils
from PIL import Image

from src.pose_estimation import load_model_from_config, load_image, estimate_poses
from src.sampling import sample_images
from src.utils import build_output

if __name__ == '__main__':

    np.random.seed(98052)
    torch.manual_seed(98052)
    torch.cuda.manual_seed(98052)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--learning_rate', type=float, default=1.0)
    parser.add_argument('--min_timestep', type=float, default=0.2)
    parser.add_argument('--max_timestep', type=float, default=0.81)
    parser.add_argument('--probe_min_timestep', type=float, default=0.2)
    parser.add_argument('--probe_max_timestep', type=float, default=0.21)
    parser.add_argument('--init_type', type=str, default='triangular', choices=['pairwise', 'triangular'])
    parser.add_argument('--optm_type', type=str, default='triangular', choices=['pairwise', 'triangular'])
    parser.add_argument('--probe_bsz', type=int, default=16)
    parser.add_argument('--adjust_iters', type=int, default=10)
    parser.add_argument('--optm_iters', type=int, default=600)
    parser.add_argument('--gen_image', action='store_true')
    parser.add_argument('--bkg_threshold', type=float, default=0.9)
    parser.add_argument('--ckpt_path', type=str, default='ckpts/105000.ckpt')
    parser.add_argument('--matcher_ckpt_path', type=str, default='ckpts/indoor_ds_new.ckpt')
    parser.add_argument('--no_est_elev', action='store_true')
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()
    print(args)

    device = 'cuda:0'
    config_path = 'src/configs/sd-objaverse-finetune-c_concat-256.yaml'

    config = OmegaConf.load(config_path)

    model = load_model_from_config(config, args.ckpt_path, device=device)
    model.eval()

    width = 256
    height = 256

    with open(args.input_json, 'r') as fin:
        jdata = json.load(fin)
        data_root = jdata['data_root']
        obj_name_list = jdata['samples']

    for obj_ent in obj_name_list[:]:

        np.random.seed(98052)
        torch.manual_seed(98052)
        torch.cuda.manual_seed(98052)
        noise = np.random.randn(args.probe_bsz, 4, 32, 32)

        if isinstance(obj_ent, tuple):
            obj_name = obj_ent[0]
            anchor_vid = obj_ent[1]
            target_vids = obj_ent[2]
            init_poses = obj_ent[3]
        elif isinstance(obj_ent, dict):
            obj_name = obj_ent['name']
            anchor_vid = obj_ent['anchor_vid']
            target_vids = obj_ent['target_vids']
            init_poses = obj_ent['init_poses']

        print(obj_name, anchor_vid, target_vids)

        name = '+'.join([str(vid) for vid in target_vids])
        name = obj_name + '_' + str(anchor_vid) + '+' + name

        obj_path = os.path.join(data_root, obj_name)

        save_root = os.path.join('outputs', args.exp_name, name)
        if not args.overwrite and os.path.exists(os.path.join(save_root, f'pose.json')):
            print('Already exists:', os.path.join(save_root, f'pose.json'), flush=True)
            continue

        os.makedirs(save_root, exist_ok=True)

        images = []

        img_path = os.path.join(obj_path, 'images', f'{anchor_vid:03d}.png')
        img = load_image(img_path, width, height, device=device, preprocessor=None, threshold=args.bkg_threshold)
        images.append(img)
        vutils.save_image((img + 1) / 2, os.path.join(save_root, f'{anchor_vid:03d}.png'))

        for vid in target_vids:
            img_path = os.path.join(obj_path, 'images', f'{vid:03d}.png')
            img = load_image(img_path, width, height, device=device, preprocessor=None, threshold=args.bkg_threshold)
            images.append(img)
            vutils.save_image((img + 1) / 2, os.path.join(save_root, f'{vid:03d}.png'))

        result_poses, aux_data = estimate_poses(
            model, images,
            learning_rate=args.learning_rate,
            init_type=args.init_type,
            optm_type=args.optm_type,
            probe_ts_range=[args.probe_min_timestep, args.probe_max_timestep],
            ts_range=[args.min_timestep, args.max_timestep],
            probe_bsz=args.probe_bsz,
            adjust_iters=args.adjust_iters,
            optm_iters=args.optm_iters,
            noise=noise,
            est_elev=not args.no_est_elev,
            matcher_ckpt_path=args.matcher_ckpt_path
        )

        if args.gen_image:
            os.makedirs(os.path.join(save_root, 'gen'), exist_ok=True)
            '''generate target images'''
            for i in range(0, len(target_vids)):
                theta, azimuth, radius = result_poses[i][0], result_poses[i][1], result_poses[i][2]
                output_imgs = sample_images(model, images[0], theta, azimuth, radius, n_samples=3)
                for oi, img in enumerate(output_imgs):
                    pil_img = Image.fromarray(img)
                    pil_img.save(os.path.join(save_root, 'gen', f'{anchor_vid:03d}_{target_vids[i]:03d}_{oi}.png'))

        jdata = build_output(anchor_vid, target_vids, result_poses, aux_data, obj_path)

        with open(os.path.join(save_root, f'pose.json'), 'w+') as fout:
            json.dump(jdata, fout, indent=4)