import sys, os, glob
import numpy as np
import re
import json
import argparse

from src.utils import relative_spherical, spherical_to_cartesian, cartesian_to_spherical, elu_to_c2w

def compute_angular_error(rotation1, rotation2):
    R_rel = rotation1.T @ rotation2
    tr = (np.trace(R_rel) - 1) / 2
    theta = np.arccos(tr.clip(-1, 1))
    return theta * 180 / np.pi


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--gt', type=str)
args = parser.parse_args()

input_root = args.input
gt_root = args.gt

name_list = os.listdir(input_root)

angles = []
angle_error_dict = {}

dists = []
position_error_dict = {}

wnt = 0

for name in name_list[:]:

    match = re.search('_[0-9]+\+[0-9+]+', name)
    if match is None:
        continue

    obj_id = name[:match.span()[0]]
    cat = obj_id.split('_')[0]

    fpath = os.path.join(input_root, name, 'pose.json')

    if not os.path.exists(fpath):
        continue

    with open(fpath, 'r') as fin:

        jdata = json.load(fin)

        avid = jdata['anchor_vid']

        obs_cnt = len(jdata['obs'])
        akey = obs_cnt

        if akey not in angle_error_dict:
            angle_error_dict[akey] = []

        if akey not in position_error_dict:
            position_error_dict[akey] = []

        anchor_c2w = np.load(os.path.join(gt_root, obj_id, 'poses', f'{avid}.npy'))
        radius = np.linalg.norm(anchor_c2w[:3, -1])

        for tvid in jdata['obs']:

            if tvid == avid:
                continue

            pred_rel_sph = np.array(jdata['obs'][tvid]['rel_sph'])

            # Scaling relative radius from the zero123 scale to the actual scale.
            # The scale range of zero123 is (1.5, 2.2), we use the average value 1.85 as the zero123 scale.
            pred_rel_sph[2] = pred_rel_sph[2] * radius / 1.85

            if os.path.exists(os.path.join(gt_root, obj_id, 'poses', f'{tvid}.npy')):
                target_c2w = np.load(os.path.join(gt_root, obj_id, 'poses', f'{tvid}.npy'))
            else:
                target_c2w = np.loadtxt(os.path.join(gt_root, obj_id, 'poses', f'{tvid}.txt'))

            gt_rel_sph = relative_spherical(target_c2w[:3, -1], anchor_c2w[:3, -1])
            gt_xyz = target_c2w[:3, -1]

            base_sph = cartesian_to_spherical(anchor_c2w[:3, -1])
            pred_xyz = spherical_to_cartesian(pred_rel_sph + base_sph)

            pred_c2w = elu_to_c2w(pred_xyz, np.zeros(3), np.array([0., 0., 1.]))
            pred_w2c = np.linalg.inv(pred_c2w)

            dist = np.linalg.norm(pred_xyz - gt_xyz, 2) / radius
            dists.append(dist)
            position_error_dict[akey].append(dist)

            # v1 = gt_xyz / np.linalg.norm(gt_xyz)
            # v2 = pred_xyz / np.linalg.norm(pred_xyz)
            # angle = np.arccos(np.dot(v1, v2)) * 180 / np.pi 

            target_rot = np.linalg.inv(target_c2w[:3, :3])
            pred_rot = pred_w2c[:3, :3]

            angle = compute_angular_error(target_rot, pred_rot)

            angles.append(angle)
            angle_error_dict[akey].append(angle)

angles = np.array(angles)

print('Angle', np.mean(angles), np.min(angles), np.max(angles), len(angles))
print('Dist', np.mean(dists), np.min(dists), np.max(dists), len(dists))

for thres in [15, 30]:

    x_angles = angles[angles < thres]
    if len(x_angles) > 0:
        print('Angle', thres, np.mean(x_angles), np.min(x_angles), np.max(x_angles), len(x_angles))
        print('RotACC', thres, float(len(x_angles)) / len(angles) * 100, len(x_angles), len(angles))

for akey in [2, 3, 4, 5, 6, 7, 8]:

    if akey not in angle_error_dict:
        continue
    
    print(akey)

    for thres in [15, 30]:
        angles = np.array(angle_error_dict[akey])
        x_angles = angles[angles < thres]
        if len(x_angles) > 0:
            print('Angle', thres, np.mean(x_angles), np.min(x_angles), np.max(x_angles))
            print('RotACC', thres, float(len(x_angles)) / len(angles) * 100, len(x_angles), len(angles))

    ces = np.array(position_error_dict[akey])
    x_ces = ces[ces < 0.2]

    print('Dist:', np.mean(ces))
    print('PosACC:', float(len(x_ces)) / len(ces) * 100)