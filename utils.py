import os, sys
import numpy as np
import math


def spherical_to_cartesian(sph):

    theta, azimuth, radius = sph

    return np.array([
        radius * np.sin(theta) * np.cos(azimuth),
        radius * np.sin(theta) * np.sin(azimuth),
        radius * np.cos(theta),
    ])


def cartesian_to_spherical(xyz):

    xy = xyz[0]**2 + xyz[1]**2
    radius = np.sqrt(xy + xyz[2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[2])
    azimuth = np.arctan2(xyz[1], xyz[0])

    return np.array([theta, azimuth, radius])


def relative_spherical(xyz_target, xyz_cond):

    sp_target = cartesian_to_spherical(xyz_target)
    sp_cond = cartesian_to_spherical(xyz_cond)

    theta_cond, azimuth_cond, z_cond = sp_cond
    theta_target, azimuth_target, z_target = sp_target

    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    d_z = z_target - z_cond

    return np.array([d_theta, d_azimuth, d_z])


def elu_to_c2w(eye, lookat, up):

    if isinstance(eye, list):
        eye = np.array(eye)
    if isinstance(lookat, list):
        lookat = np.array(lookat)
    if isinstance(up, list):
        up = np.array(up)

    l = eye - lookat
    l = l / np.linalg.norm(l)
    s = np.cross(l, up)
    s = s / np.linalg.norm(s)
    uu = np.cross(s, l)

    rot = np.eye(3)
    rot[0, :] = -s
    rot[1, :] = uu
    rot[2, :] = l
    
    c2w = np.eye(4)
    c2w[:3, :3] = rot.T
    c2w[:3, 3] = eye

    return c2w


def c2w_to_elu(c2w):

    w2c = np.linalg.inv(c2w)
    eye = c2w[:3, 3]
    lookat_dir = -w2c[2, :3]
    lookat = eye + lookat_dir
    up = w2c[1, :3]

    return eye, lookat, up


def build_output(anchor_vid, target_vids, pred_sphs, aux_data, obj_root, export_xyz=False):

    '''save pose output'''
    jdata = {
        'anchor_vid': anchor_vid,
        'obs': {}
    }

    jdata['obs'][anchor_vid] = {
        'img_path': f'{anchor_vid:03d}.png'
    }

    for key in aux_data:
        jdata['obs'][anchor_vid][key] = aux_data[key][0]

    anchor_sph = None

    if os.path.exists(os.path.join(obj_root, 'poses', f'{anchor_vid:03d}.npy')):

        anchor_rt = np.load(os.path.join(obj_root, 'poses', f'{anchor_vid:03d}.npy'))
        anchor_xyz = anchor_rt[:3, -1]

        anchor_sph = cartesian_to_spherical(anchor_xyz)
        jdata['obs'][anchor_vid]['sph'] = anchor_sph.tolist()

        if export_xyz:        
            jdata['obs'][anchor_vid]['xyz'] = {
                'x': anchor_xyz[0],
                'y': anchor_xyz[1],
                'z': anchor_xyz[2]
            }


    for i in range(0, len(target_vids)):
        
        target_vid = target_vids[i]

        rel_sph = np.array(pred_sphs[i])

        opack = {
            'img_path': f'{target_vid:03d}.png',
            'rel_sph': rel_sph.tolist()
        }

        for key in aux_data:
            opack[key] = aux_data[key][i+1]

        if anchor_sph is not None:

            target_sph = anchor_sph + rel_sph

            if export_xyz:
                target_xyz = spherical_to_cartesian(target_sph)

                opack['xyz'] = {
                    'x': target_xyz[0],
                    'y': target_xyz[1],
                    'z': target_xyz[2]
                }

            if os.path.exists(os.path.join(obj_root, 'poses', f'{target_vid:03d}.npy')):

                target_rt = np.load(os.path.join(obj_root, 'poses', f'{target_vid:03d}.npy'))

                if export_xyz:
                    opack['gt_xyz'] = {
                        'x': target_rt[0, -1],
                        'y': target_rt[1, -1],
                        'z': target_rt[2, -1]
                    }

                gt_rel_sph = relative_spherical(target_rt[:3, -1], anchor_rt[:3, -1])

                opack['gt_rel_sph'] = gt_rel_sph.tolist()

        jdata['obs'][target_vid] = opack

    return jdata