import os, sys
import numpy as np
import math
import rembg
import cv2

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

    jdata['obs'][anchor_vid] = {}

    for key in aux_data:
        jdata['obs'][anchor_vid][key] = aux_data[key][0]

    anchor_sph = None

    anchor_rt = None

    if os.path.exists(os.path.join(obj_root, 'poses', f'{anchor_vid}.npy')):
        anchor_rt = np.load(os.path.join(obj_root, 'poses', f'{anchor_vid}.npy'))
    elif os.path.exists(os.path.join(obj_root, 'poses', f'{anchor_vid}.txt')):
        anchor_rt = np.loadtxt(os.path.join(obj_root, 'poses', f'{anchor_vid}.txt'))

    if anchor_rt is not None:

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

            target_rt = None

            if os.path.exists(os.path.join(obj_root, 'poses', f'{target_vid}.npy')):
                target_rt = np.load(os.path.join(obj_root, 'poses', f'{target_vid}.npy'))
            elif os.path.exists(os.path.join(obj_root, 'poses', f'{target_vid}.txt')):
                target_rt = np.loadtxt(os.path.join(obj_root, 'poses', f'{target_vid}.txt'))

            if target_rt is not None:

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


def remove_background(image, rembg_session = None, force = False, **rembg_kwargs):
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image


def group_cropping(images, width, height, ratio=1.5, mask_thres=127, bkg_color=[255, 255, 255, 255]):

    ws = []
    hs = []

    images = [ np.asarray(img) for img in images ]

    for img in images:

        alpha = img[:, :, 3]

        yy, xx = np.where(alpha > mask_thres)
        y0, y1 = yy.min(), yy.max()
        x0, x1 = xx.min(), xx.max()

        ws.append(x1 - x0)
        hs.append(y1 - y0)

    sz_w = np.max(ws)
    sz_h = np.max(hs)

    sz = int( max(ratio*sz_w, ratio*sz_h) )

    out_rgbs = []

    for rgba in images:

        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]

        yy, xx = np.where(alpha > mask_thres)
        y0, y1 = yy.min(), yy.max()
        x0, x1 = xx.min(), xx.max()

        height, width, chn = rgb.shape

        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
  
        y0 = cy - int(np.floor(sz / 2))
        y1 = cy + int(np.ceil(sz / 2))
        x0 = cx - int(np.floor(sz / 2))
        x1 = cx + int(np.ceil(sz / 2))
        out = rgba[ max(y0, 0) : min(y1, height) , max(x0, 0) : min(x1, width), : ].copy()
        pads = [(max(0-y0, 0), max(y1-height, 0)), (max(0-x0, 0), max(x1-width, 0)), (0, 0)]
        out = np.pad(out, pads, mode='constant', constant_values=0)

        assert(out.shape[:2] == (sz, sz))

        out[:, :, :3] = out[:, :, :3] * (out[..., 3:]/255.) + np.array(bkg_color)[None, None, :3] * (1-out[..., 3:]/255.)
        out[:, :, -1] = bkg_color[-1]

        out = cv2.resize(out.astype(np.uint8), (256, 256))
        out = out[:, :, :3]

        out_rgbs.append(out)

    return out_rgbs