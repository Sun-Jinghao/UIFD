import warnings
warnings.filterwarnings(action='once')
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import cv2
import numpy as np
import torch
import torch.nn.functional as func


#from .pytorch_utils import keypoints_to_grid
def keypoints_to_grid(keypoints, img_size):
    """
    Convert a tensor [N, 2] or batched tensor [B, N, 2] of N keypoints into
    a grid in [-1, 1]² that can be used in torch.nn.functional.interpolate.
    """
    n_points = keypoints.size()[-2]
    device = keypoints.device
    grid_points = keypoints.float() * 2. / torch.tensor(
        img_size, dtype=torch.float, device=device) - 1.
    grid_points = grid_points[..., [1, 0]].view(-1, n_points, 1, 2) #B*np*1*2
    return grid_points

# from ..datasets.utils.homographies import warp_points
def warp_points(points, H):
    """
    Warp 2D points by an homography H.
    """
    n_points = points.shape[0]
    reproj_points = points.copy()[:, [1, 0]]
    reproj_points = np.concatenate([reproj_points, np.ones((n_points, 1))],
                                   axis=1)
    reproj_points = H.dot(reproj_points.transpose()).transpose()
    reproj_points = reproj_points[:, :2] / reproj_points[:, 2:]
    reproj_points = reproj_points[:, [1, 0]]
    return reproj_points

def keep_true_keypoints(points, H, shape, margin=0):
    """ Keep only the points whose warped coordinates by H are still
        inside shape, possibly within a margin to the border. """
    warped_points = warp_points(points, H)
    mask = ((warped_points[:, 0] >= margin)
            & (warped_points[:, 0] < shape[0] - margin)
            & (warped_points[:, 1] >= margin)
            & (warped_points[:, 1] < shape[1] - margin))
    return points[mask, :], mask


def select_k_best(points, scores, k, H=None, shape=None, margin=0):
    """ Select the k best scoring points. If H and shape are defined,
        additionally keep only the keypoints that once warped by H are
        still inside shape within a given margin to the border. """
    if H is None:
        mask = np.zeros_like(scores, dtype=bool)
        mask[scores.argsort()[-k:]] = True
    else:
        true_points, mask = keep_true_keypoints(points, H, shape, margin)
        filtered_scores = scores.copy()
        filtered_scores[~mask] = 0
        rank_mask = np.zeros_like(scores, dtype=bool)
        rank_mask[filtered_scores.argsort()[-k:]] = True
        mask = mask & rank_mask
    return points[mask], mask


def get_lisrd_desc_dist(descs, meta_descs):
    """ Get a descriptor distance,
        weighted by the similarity of meta descriptors. """
    desc_dists, meta_desc_sims = [], []
    for i in range(len(descs)):
        desc_dists.append(2 - 2 * (descs[i][0] @ descs[i][1].t()))
        meta_desc_sims.append(meta_descs[i][0] @ meta_descs[i][1].t())
    desc_dists = torch.stack(desc_dists, dim=2)
    meta_desc_sims = torch.stack(meta_desc_sims, dim=2)

    # Weight the descriptor distances
    meta_desc_sims = func.softmax(meta_desc_sims, dim=2)
    desc_dists = torch.sum(desc_dists * meta_desc_sims, dim=2)

    return desc_dists


def get_lisrd_desc_dist_numpy(desc0, desc1, meta_desc0, meta_desc1):
    """ Get a descriptor distance in numpy,
        weighted by the similarity of meta descriptors. """
    desc_dists, meta_desc_sims = [], []
    for i in range(desc0.shape[1]):
        desc_dists.append(2 - 2 * (desc0[:, i] @ desc1[:, i].transpose()))
        meta_desc_sims.append(meta_desc0[:, i] @ meta_desc1[:, i].transpose())
    desc_dists = np.stack(desc_dists, axis=2)
    meta_desc_sims = np.stack(meta_desc_sims, axis=2)

    # Weight the descriptor distances
    meta_desc_sims = np.exp(meta_desc_sims)
    meta_desc_sims /= np.sum(meta_desc_sims, axis=2, keepdims=True)
    desc_dists = np.sum(desc_dists * meta_desc_sims, axis=2)

    return desc_dists


def extract_descriptors(keypoints, descriptors, meta_descriptors, img_size):
    """ Sample descriptors and meta descriptors at keypoints positions.
        This assumes a batch_size of 1. """
    grid_points = keypoints_to_grid(keypoints, img_size)

    # Extract the local descriptors
    descs = []
    for k in descriptors.keys():
        desc = func.normalize(func.grid_sample(descriptors[k], grid_points),
                              dim=1)[0, :, :, 0].t()
        descs.append(desc)
    descs = torch.stack(descs, dim=1)

    # Extract the meta descriptors
    meta_descs = []
    for k in meta_descriptors.keys():
        meta_desc = func.normalize(
            func.grid_sample(meta_descriptors[k], grid_points),
            dim=1)[0, :, :, 0].t()
        meta_descs.append(meta_desc)
    meta_descs = torch.stack(meta_descs, dim=1)

    return descs, meta_descs


def lisrd_matcher(desc1, desc2, meta_desc1, meta_desc2):
    """ Nearest neighbor matcher for LISRD. """
    device = desc1.device
    desc_weights = torch.einsum('nid,mid->nim', (meta_desc1, meta_desc2))
    del meta_desc1, meta_desc2
    desc_weights = func.softmax(desc_weights, dim=1)
    desc_sims = torch.einsum('nid,mid->nim', (desc1, desc2)) * desc_weights
    del desc1, desc2, desc_weights
    desc_sims = torch.sum(desc_sims, dim=1)
    nn12 = torch.max(desc_sims, dim=1)[1]
    nn21 = torch.max(desc_sims, dim=0)[1]
    ids1 = torch.arange(desc_sims.shape[0], dtype=torch.long, device=device)
    del desc_sims
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]], dim=1)
    return matches

def sample_descriptors(keypoints, descriptors, img_size):
    """ Sample descriptors at keypoints positions.
        This assumes a batch_size of 1. """
    grid_points = keypoints_to_grid(keypoints, img_size) # B*np*1*2

    # Extract the local descriptors #descriptors[k] B*C*H*W
    descs = []
    for k in descriptors.keys():
        desc = func.normalize(func.grid_sample(descriptors[k], grid_points),
                              dim=1)[0, :, :, 0].t() # np*C
        descs.append(desc)
    descs = torch.stack(descs, dim=1) #np*1*C #有几个描述子，中间维度就几个
   #  descs = descs[:,0,...]
    return descs

def mvnet_matcher(desc1, desc2):
    """ Nearest neighbor matcher. """
    device = desc1.device
    desc_sims = torch.einsum('nid,mid->nim', (desc1, desc2)) #n*1*m
    del desc1, desc2
    desc_sims = torch.sum(desc_sims, dim=1) #n*m
    nn12 = torch.max(desc_sims, dim=1)[1]#[1]只取出索引不取值
    nn21 = torch.max(desc_sims, dim=0)[1]
    ids1 = torch.arange(desc_sims.shape[0], dtype=torch.long, device=device)
    del desc_sims
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]], dim=1)
    return matches


def filter_outliers_ransac(kp1, kp2):
    """ Given pairs of candidate matches, filter them
        based on homography fitting with RANSAC. """
    inliers = cv2.findHomography(kp1, kp2, cv2.RANSAC)[1][:, 0].astype(bool)
    return kp1[inliers], kp2[inliers]
