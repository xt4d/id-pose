import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from datetime import datetime

from .ldm.util import load_and_preprocess, instantiate_from_config
from .pose_funcs import probe_pose, find_optimal_poses, get_inv_pose, add_pose, pairwise_loss

from .oee.utils.elev_est_api import elev_est_api, ElevEstHelper
from .sampling import sample_images


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
        noise=None,
        est_elev=False,
        matcher_ckpt_path=None
    ):

    num = len(images)

    if num <= 2:
        init_type = 'pairwise'

    cands = {}
    losses = {}

    elevs = {i: None for i in range(num)}
    elev_ranges = {i: None for i in range(num)}

    init_poses = {i: None for i in range(num)}
    pairwise_init_poses = {i: None for i in range(num)}

    print('Initialization: Probe', datetime.now())

    if est_elev:
        matcher = ElevEstHelper.get_feature_matcher(matcher_ckpt_path, model.device)
        for i in range(num):
            simgs = sample_surrounding_images(model, images[i])
            elev = elev_est_api(matcher, simgs, min_elev=20, max_elev=160)
            elevs[i] = elev
        
        for i in range(num):
            if elevs[i] is not None:
                elevs[i] = np.deg2rad(elevs[i])

        for i in range(1, num):

            if elevs[i] is not None and elevs[0] is not None:
                elev_ranges[i] = np.array([ elevs[i] - elevs[0] ])
            elif elevs[i] is not None:
                elev_ranges[i] = -make_elev_probe_range(elevs[i])
            elif elevs[0] is not None:
                elev_ranges[i] = make_elev_probe_range(elevs[0])

    images = [ img.permute(0, 2, 3, 1) for img in images ]

    for i in range(1, num):

        print('PAIR', 0, i, datetime.now())

        all_cands = probe_pose(model, images[0], images[i], probe_ts_range, probe_bsz, theta_range=elev_ranges[i], noise=noise)

        print('Adjust candidates', len(all_cands), datetime.now())

        adjusted_cands = all_cands[:5]
        if adjust_iters > 0:
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
                adjusted_cands.append((loss, out_poses[0], cand[0], cand[1]))

            adjusted_cands = sorted(adjusted_cands)[:5]

        for cand in adjusted_cands:
            print(cand)

        cands[i] = [ cand[:2] for cand in adjusted_cands ]
        losses[i] = [loss if (init_type == 'pairwise') else 0.0 for loss, _ in cands[i]]

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

    aux_data = {
        'tri_init_sph': init_poses,
        'pw_init_sph': pairwise_init_poses,
        'elev': elevs
    }

    return out_poses, aux_data


def make_elev_probe_range(elev, interval=np.pi/4):

    up_range = np.arange(elev, 0, -interval)
    down_range = np.arange(elev+interval, np.pi, interval)
    probe_range = np.concatenate([up_range, down_range])
    probe_range -= elev

    return probe_range


def sample_surrounding_images(model, image):

    s0 = sample_images(model, image, float(np.deg2rad(-10)), 0, 0, n_samples=1)
    s1 = sample_images(model, image, float(np.deg2rad(+10)), 0, 0, n_samples=1)
    s2 = sample_images(model, image, 0, float(np.deg2rad(-10)), 0, n_samples=1)
    s3 = sample_images(model, image, 0, float(np.deg2rad(+10)), 0, n_samples=1)

    return s0 + s1 + s2 + s3