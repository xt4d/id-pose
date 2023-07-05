import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from datetime import datetime

from ldm.util import load_and_preprocess, instantiate_from_config
from pose_funcs import probe_pose, find_optimal_poses, get_inv_pose, add_pose, pairwise_loss


def load_image(img_path, width, height, mask_path=None, device='cpu', preprocessor=None, threshold=0.9):

    img = Image.open(img_path)

    if preprocessor is not None:
        img = load_and_preprocess(preprocessor, img)
    else:
        if img.mode == 'RGBA':
            img = np.asarray(img, dtype=np.float32) / 255.
            img[img[:, :, -1] <= threshold] = [1., 1., 1., 1.] # thresholding background
            img = img[:, :, :3]
        elif img.mode == 'RGB':
            if mask_path is not None:
                mask = Image.open(mask_path)
                bkg = Image.new('RGB', (width, height), color=(255, 255, 255))
                img = Image.composite(img, bkg, mask)
            img = np.asarray(img, dtype=np.float32) / 255.
        else:
            print('Wrong format:', img_path)

    img = transforms.ToTensor()(img).unsqueeze(0).to(device)
    img = img * 2 - 1
    img = transforms.functional.resize(img, [height, width])

    return img


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location=device)
    if 'global_step' in pl_sd:
        step = pl_sd['global_step']
        print(f'Global Step: {step}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


def estimate_poses(
        model, images, learning_rate, 
        init_type='pairwise', 
        optm_type='pairwise', 
        probe_ts_range=[0.02, 0.98], ts_range=[0.02, 0.98], 
        probe_bsz=16, 
        adjust_iters=10, optm_iters=600, 
        noise=None
    ):

    num = len(images)

    if num <= 2:
        init_type = 'pairwise'

    cands = {}
    losses = {}

    pairwise_init_poses = {}

    print('Initialization: Probe', datetime.now())

    for i in range(1, num):

        print('PAIR', 0, i, datetime.now())

        all_cands = probe_pose(model, images[0], images[i], probe_ts_range, probe_bsz, noise=noise)

        print('Adjust candidates', len(all_cands), datetime.now())

        adjusted_cands = []
        '''only adjust the first half'''
        for cand in all_cands[:len(all_cands)//2]:
            
            out_poses, _, _ = find_optimal_poses(
                model, [images[0], images[i]], 
                learning_rate*10, n_iter=adjust_iters, 
                init_poses={1: cand[1]}, 
                ts_range=ts_range,
                print_n=100,
                avg_last_n=1
            )

            loss = pairwise_loss(out_poses[0], model, images[0], images[i], probe_ts_range, probe_bsz, noise=noise)
            adjusted_cands.append((loss, out_poses[0]))

        adjusted_cands = sorted(adjusted_cands)[:5]
        for cand in adjusted_cands:
            print(cand)

        cands[i] = adjusted_cands
        losses[i] = [loss if (init_type == 'pairwise') else 0.0 for loss, _ in adjusted_cands]

        pairwise_init_poses[i] = min(cands[i])[1]

    print('Initialization: Select', datetime.now())

    if init_type == 'triangular':

        for i in range(1, num):

            for j in range(i+1, num):

                iloss = [ [None for v in range(0, len(cands[j]))] for u in range(0, len(cands[i])) ]
                jloss = [ [None for u in range(0, len(cands[i]))] for v in range(0, len(cands[j])) ]

                for u in range(0, len(cands[i])):

                    la, pa = cands[i][u]

                    # pose i -> 0
                    pa = get_inv_pose(pa)

                    for v in range(0, len(cands[j])):

                        # pose 0 -> j
                        lb, pb = cands[j][v]

                        theta, azimuth, radius = add_pose(pa, pb)
                        lp = pairwise_loss([theta, azimuth, radius], model, images[i], images[j], probe_ts_range, probe_bsz, noise=noise)

                        iloss[u][v] = la + lb + lp
                        jloss[v][u] = la + lb + lp

                for u in range(0, len(cands[i])):
                    losses[i][u] += min(min(iloss[u]), cands[i][u][0]*3)
                
                for v in range(0, len(cands[j])):
                    losses[j][v] += min(min(jloss[v]), cands[j][v][0]*3)

    init_poses = {}

    for i in range(1, num):

        ranks = sorted([x for x in range(0, len(losses[i]))], key=lambda x: losses[i][x])

        min_rank = ranks[0]

        for u in range(0, len(cands[i])):
            print(cands[i][u], losses[i][u])
        print(i, 'SELECT', min_rank, losses[i][min_rank])

        init_poses[i] = cands[i][min_rank][1]

    print('Optimization', datetime.now())

    combinations = None
    if optm_type == 'pairwise':
        combinations = [ (0, i) for i in range(1, num) ] + [ (i, 0) for i in range(1, num) ]

    elif optm_type == 'triangular':
        combinations = []
        for i in range(0, num):
            for j in range(i+1, num):
                combinations.append((i, j))
                combinations.append((j, i))

    print('Combinations', len(combinations), combinations)

    '''Optimization'''
    out_poses, _, loss = find_optimal_poses(
        model, images, 
        learning_rate, n_iter=(num-1)*optm_iters, 
        init_poses=init_poses, 
        ts_range=ts_range,
        combinations=combinations,
        avg_last_n=20,
        print_n=100
    )

    print('Done', datetime.now())

    return out_poses, [ init_poses[i] for i in range(1, num) ], [ pairwise_init_poses[i] for i in range(1, num) ]
