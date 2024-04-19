import numpy as np
import torch

class PoseT(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pose):

        p1 = pose[..., 0:1]
        p2 = torch.sin(pose[..., 1:2])
        p3 = torch.cos(pose[..., 1:2])
        p4 = pose[..., 2:]

        return torch.cat([p1, p2, p3, p4], dim=-1)


@torch.no_grad()
def noise_loss(model, cond_image, target_image, pose, ts_range, bsz, noise=None):

    mx = ts_range[1]
    mn = ts_range[0]

    pose_layer = PoseT()

    batch = {}
    batch['image_target'] = target_image.repeat(bsz, 1, 1, 1)
    batch['image_cond'] = cond_image.repeat(bsz, 1, 1, 1)
    batch['T'] = pose_layer(pose.detach()).repeat(bsz, 1)

    if noise is not None:
        noise = torch.tensor(noise, dtype=model.dtype, device=model.device)

    loss, _ = model.shared_step(batch, ts=np.arange(mn, mx, (mx-mn) / bsz), noise=noise[:bsz])

    return loss.item()


@torch.no_grad()
def pairwise_loss(pose, model, cond_image, target_image, ts_range, probe_bsz, noise=None):

    theta, azimuth, radius = pose

    pose1 = torch.tensor([[theta, azimuth, radius]], device=model.device, dtype=torch.float32)
    pose2 = torch.tensor([[-theta, np.pi*2-azimuth, -radius]], device=model.device, dtype=torch.float32)
    loss1 = noise_loss(model, cond_image, target_image, pose1, ts_range, probe_bsz, noise=noise)
    loss2 = noise_loss(model, target_image, cond_image, pose2, ts_range, probe_bsz, noise=noise)

    return loss1 + loss2


@torch.no_grad()
def probe_pose(model, cond_image, target_image, ts_range, probe_bsz, theta_range=None, azimuth_range=None, radius_range=None, noise=None):

    eps = 1e-5

    if theta_range is None:
        theta_range = np.arange(start=-np.pi*2/3, stop=np.pi*2/3+eps, step=np.pi/3)
    if azimuth_range is None:
        azimuth_range = np.arange(start=0.0, stop=np.pi*2, step=np.pi/4)
    if radius_range is None:
        radius_range = np.arange(start=0.0, stop=0.0+eps, step=0.1)

    cands = []

    for radius in radius_range:
        for azimuth in azimuth_range:
            for theta in theta_range:

                loss = pairwise_loss([theta, azimuth, radius], model, cond_image, target_image, ts_range, probe_bsz, noise=noise)

                '''convert numpy.float to float'''
                cands.append((loss, [float(theta), float(azimuth), float(radius)]))

    return cands


def create_random_pose():

    theta = np.random.rand() * np.pi - np.pi / 2
    azimuth = np.random.rand() * np.pi * 2
    radius = np.random.rand() - 0.5
    
    return [theta, azimuth, radius]


def get_inv_pose(pose):

    return [-pose[0], np.pi*2 - pose[1], -pose[2]]


def add_pose(pose1, pose2):

    theta = pose1[0] + pose2[0]
    azimuth = pose1[1] + pose2[1]
    azimuth = azimuth % (np.pi*2)

    return [ theta, azimuth, (pose1[2] + pose2[2]) ]


def create_pose_params(pose, device):

    theta = torch.tensor([pose[0]], requires_grad=True, device=device)
    azimuth = torch.tensor([pose[1]], requires_grad=True, device=device)
    radius = torch.tensor([pose[2]], requires_grad=True, device=device)

    return [theta, azimuth, radius]


def find_optimal_poses(model, images, learning_rate, bsz=1, n_iter=1000, init_poses={}, ts_range=[0.02, 0.92], combinations=None, print_n=50, avg_last_n=1):
    
    layer = PoseT()

    num = len(images)

    batch = {}

    pose_params = { i:None for i in range(1, num)}
    pose_trajs = { i:[]  for i in range(1, num) }

    for i in range(1, num):

        if i in init_poses:
            init_pose = init_poses[i]
        else:
            init_pose = create_random_pose()

        pose = create_pose_params(init_pose, model.device)
        pose_params[i] = pose

    if combinations is None:
        combinations = []
        for i in range(0, num):
            for j in range(i+1, num):
                combinations.append((i, j))
                combinations.append((j, i))

    param_list = []
    for i in pose_params:
        param_list += pose_params[i]

    optimizer = torch.optim.SGD(param_list, lr = learning_rate)

    loss_traj = []
    select_indces = set([])
                                
    for iter in range(0, n_iter):

        if print_n > 0 and iter % print_n == 0 and iter > 0:
            print(iter, np.mean(loss_traj[-avg_last_n:]), flush=True)
            for i in range(1, num):
                print(0, i, np.mean(pose_trajs[i][-avg_last_n:], axis=0).tolist())

        '''record poses'''
        for i in select_indces:
            pose = pose_params[i]
            pose_trajs[i].append([pose[0].item(), pose[1].item(), pose[2].item()])

        select_indces = set([])

        conds = []
        targets = []
        rts = []

        choices = [ iter % len(combinations) ] 
        
        if bsz > 1:
            choices = np.random.choice(len(combinations), size=bsz, replace=True)

        for cho in choices:

            i, j = combinations[cho]

            conds.append(images[i])
            targets.append(images[j])
            if i == 0:
                pose = pose_params[j]
                select_indces.add(j)
            
            elif j == 0:
                pose = get_inv_pose(pose_params[i])
                select_indces.add(i)

            else:
                pose0j = pose_params[j]
                posei0 = get_inv_pose(pose_params[i])

                if np.random.rand() < 0.5:
                    posei0 = [a.item() for a in posei0]
                    select_indces.add(j)
                else:
                    pose0j = [b.item() for b in pose0j]
                    select_indces.add(i)

                #pose = [ torch.remainder(a+b+2*np.pi, 2*np.pi) - np.pi for a, b in zip(posei0, pose0j) ]
                pose = [ a+b for a, b in zip(posei0, pose0j) ]

            rts.append(torch.cat(pose)[None, ...])

        batch['image_cond'] = torch.cat(conds, dim=0)
        batch['image_target'] = torch.cat(targets, dim=0)
        batch['T'] = layer(torch.cat(rts, dim=0))
        ts = np.arange(ts_range[0], ts_range[1], (ts_range[1]-ts_range[0]) / len(conds))

        optimizer.zero_grad()
        loss, loss_dict = model.shared_step(batch, ts=ts)
        loss.backward()

        optimizer.step()

        loss_traj.append(loss.item())

    if n_iter > 0:
        result_poses = [ np.mean(pose_trajs[i][-avg_last_n:], axis=0).tolist() for i in range(1, num) ]
        result_loss = np.mean(loss_traj[-avg_last_n:])
    else:
        result_poses = [ init_poses[i] for i in range(1, num) ]
        result_loss = None

    return result_poses, [ init_poses[i] for i in range(1, num) ], result_loss
