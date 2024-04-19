import os, sys
import argparse
import json

import torch
import numpy as np
from omegaconf import OmegaConf
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
import rembg

from src.pose_estimation import load_model_from_config, load_image, estimate_poses, estimate_elevs
from src.sampling import sample_images
from src.utils import build_output, remove_background, group_cropping

if __name__ == '__main__':

    np.random.seed(98052)
    torch.manual_seed(98052)
    torch.cuda.manual_seed(98052)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--min_timestep', type=float, default=0.2)
    parser.add_argument('--max_timestep', type=float, default=0.21)
    parser.add_argument('--seed_cand_num', type=int, default=8)
    parser.add_argument('--probe_min_timestep', type=float, default=0.2)
    parser.add_argument('--probe_max_timestep', type=float, default=0.21)
    parser.add_argument('--explore_type', type=str, default='triangular', choices=['pairwise', 'triangular'])
    parser.add_argument('--refine_type', type=str, default='triangular', choices=['pairwise', 'triangular'])
    parser.add_argument('--probe_bsz', type=int, default=16)
    parser.add_argument('--adjust_factor', type=float, default=10.0)
    parser.add_argument('--adjust_iters', type=int, default=10)
    parser.add_argument('--adjust_bsz', type=int, default=1)
    parser.add_argument('--refine_factor', type=float, default=1.0)
    parser.add_argument('--refine_iters', type=int, default=600)
    parser.add_argument('--refine_bsz', type=int, default=1)
    parser.add_argument('--gen_image', action='store_true')
    parser.add_argument('--no_rembg', action='store_true')
    parser.add_argument('--bkg_threshold', type=float, default=0.9)
    parser.add_argument('--ckpt_path', type=str, default='ckpts/zero123-xl.ckpt')
    parser.add_argument('--matcher_ckpt_path', type=str, default='ckpts/indoor_ds_new.ckpt')
    parser.add_argument('--est_elev', type=str, default='all')
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

    ###### Process Input ######
    if os.path.isdir(args.input):

        data_root = args.input
        if data_root.endswith('/'):
            data_root = data_root[:-1]
        inames = [ fname for fname in sorted(os.listdir(data_root)) if (fname.lower().endswith('.png') or fname.lower().endswith('.jpg') or fname.lower().endswith('.webp')) ]
        vids = [ iname.split('.')[0] for iname in inames ]
        img_paths = { iname.split('.')[0] : os.path.join(data_root, iname) for iname in inames }

        obj_ent = {
            'name': os.path.basename(data_root),
            'anchor_vid': vids[0],
            'target_vids': vids[1:],
            'img_paths': img_paths,
            'init_poses': {}
        }
        obj_name_list = [ obj_ent ]

    elif args.input.endswith('.json'):
        with open(args.input, 'r') as fin:
            jdata = json.load(fin)
            data_root = jdata['data_root']
            obj_name_list = jdata['samples']
    else:
        assert f'Invalid Input: {args.input}'

    ###### Estimation for each Entity ######
    for obj_ent in obj_name_list[:]:

        np.random.seed(98052)
        torch.manual_seed(98052)
        torch.cuda.manual_seed(98052)
        noise = np.random.randn(args.probe_bsz, 4, 32, 32)

        obj_name = obj_ent['name']
        anchor_vid = obj_ent['anchor_vid']
        target_vids = obj_ent['target_vids']
        init_poses = obj_ent['init_poses']
        img_paths = obj_ent['img_paths'] if 'img_paths' in obj_ent else {}

        anchor_vid = f'{anchor_vid:03d}' if isinstance(anchor_vid, int) else anchor_vid
        target_vids = [ f'{vid:03d}' if isinstance(vid, int) else vid for vid in target_vids ]

        print(obj_name, anchor_vid, target_vids)

        name = '+'.join([vid for vid in target_vids])
        name = obj_name + '_' + anchor_vid + '+' + name

        obj_path = os.path.join(data_root, obj_name)

        save_root = os.path.join(args.output, name)
        if not args.overwrite and os.path.exists(os.path.join(save_root, f'pose.json')):
            print('Already exists:', os.path.join(save_root, f'pose.json'), flush=True)
            continue

        os.makedirs(save_root, exist_ok=True)

        if args.no_rembg:
            np_images = []
            for vid in [anchor_vid] + target_vids:
                img_path = img_paths[vid] if vid in img_paths else os.path.join(obj_path, 'images', f'{vid}.png')
                img = load_image(img_path, threshold=args.bkg_threshold)
                np_images.append(img)
        else:
            rembg_session = rembg.new_session()
            images = []
            for vid in [anchor_vid] + target_vids:
                img_path = img_paths[vid] if vid in img_paths else os.path.join(obj_path, 'images', f'{vid}.png')
                img = Image.open(img_path)
                img = remove_background(img, rembg_session=rembg_session, force=True)
                images.append(img)
            np_images = group_cropping(images, width, height)

        images = []

        for vid, img in zip([anchor_vid] + target_vids, np_images):
            img = transforms.ToTensor()(img).unsqueeze(0).to(device)
            img = img * 2 - 1
            img = transforms.functional.resize(img, [height, width])
            images.append(img)
            vutils.save_image((img + 1) / 2, os.path.join(save_root, f'{vid}.png'))

        elevs, elev_ranges = estimate_elevs(
            model, images, est_type=args.est_elev, matcher_ckpt_path=args.matcher_ckpt_path
        )

        result_poses, aux_data = estimate_poses(
            model, images,
            seed_cand_num=args.seed_cand_num,
            explore_type=args.explore_type,
            refine_type=args.refine_type,
            probe_ts_range=[args.probe_min_timestep, args.probe_max_timestep],
            ts_range=[args.min_timestep, args.max_timestep],
            probe_bsz=args.probe_bsz,
            adjust_factor=args.adjust_factor,
            adjust_iters=args.adjust_iters,
            adjust_bsz=args.adjust_bsz,
            refine_factor=args.refine_factor,
            refine_iters=args.refine_iters,
            refine_bsz=args.refine_bsz,
            noise=noise,
            elevs=elevs,
            elev_ranges=elev_ranges
        )

        if args.gen_image:
            os.makedirs(os.path.join(save_root, 'gen'), exist_ok=True)
            '''generate target images'''
            for i in range(0, len(target_vids)):
                theta, azimuth, radius = result_poses[i][0], result_poses[i][1], result_poses[i][2]
                output_imgs = sample_images(model, images[0], theta, azimuth, radius, n_samples=3)
                for oi, img in enumerate(output_imgs):
                    pil_img = Image.fromarray(img)
                    pil_img.save(os.path.join(save_root, 'gen', f'{anchor_vid}_{target_vids[i]}_{oi}.png'))

        jdata = build_output(anchor_vid, target_vids, result_poses, aux_data, obj_path)

        with open(os.path.join(save_root, f'pose.json'), 'w+') as fout:
            json.dump(jdata, fout, indent=4)
