import cv2
import numpy as np
import torch
import torch.nn.functional as func
from model import get_model
from model.base_model import Mode
from model.keypoint_detectors import SP_detect, load_SP_net
from utils1 import sample_descriptors, mvnet_matcher, filter_outliers_ransac

def my_drawMatches(img1, matched_kp1, img2, matched_kp2):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    #out[:rows1, :cols1] = np.dstack([img1, img1, img1])
    #out[:rows2, cols1:] = np.dstack([img2, img2, img2])
    out[:rows1, :cols1] = img1.copy()
    out[:rows2, cols1:] = img2.copy()
    num_matches = len(matched_kp1)
    for i in range(num_matches):
        (x1, y1) = matched_kp1[i]
        (x2, y2) = matched_kp2[i]
        cv2.circle(out, (int(np.round(x1)), int(np.round(y1))), 1, (255, 0, 0), 1)
        cv2.circle(out, (int(np.round(x2) + cols1), int(np.round(y2))), 1, (255, 0, 0), 1)
        cv2.line(out, (int(np.round(x1)), int(np.round(y1))), (int(np.round(x2) + cols1), int(np.round(y2))), (0, 255, 0), 1)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out

if __name__ == "__main__":

    # load images
    img1 = cv2.cvtColor(cv2.imread('/home/ouc/RAY/test/assets/rdnim_samples/day.jpg'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('/home/ouc/RAY/test/assets/rdnim_samples/dusk.jpg'), cv2.COLOR_BGR2RGB)
    img_size_ori = img1.shape[:2]  # height width
    # resize images
    dim = (320, 240) # width height
    img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_CUBIC)
    img_size = img1.shape[:2]  # height width

    # Define Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the DeLoF-VoTer model
    checkpoint_path = '/home/ouc/RAY/test/weights/fastmvnet1.pth'
    model_config = {'name': 'lisrd', 'desc_size': 128, 'tile': 3, 'n_clusters': 8, 'meta_desc_dim': 128,
                    'learning_rate': 0.001, 'compute_meta_desc': False, 'freeze_local_desc': False}
    my_net = get_model('lisrd')(None, model_config, device)
    my_net.load(checkpoint_path, Mode.EXPORT)
    my_net._net.eval()

    # Load the keypoint model, here SuperPoint #内部指定了路径载入model
    kp_net = load_SP_net(conf_thresh=0.015, cuda=torch.cuda.is_available(), nms_dist=4, nn_thresh=0.7)

    with torch.no_grad():
        # Keypoint detection
        kp1 = SP_detect(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), kp_net)
        gpu_kp1 = torch.tensor(kp1, dtype=torch.float, device=device)[:, :2]
        kp2 = SP_detect(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), kp_net)
        gpu_kp2 = torch.tensor(kp2, dtype=torch.float, device=device)[:, :2]

        # Descriptor inference #opencv h w c
        gpu_img1 = torch.tensor(img1, dtype=torch.float, device=device)
        inputs1 = {'image0': gpu_img1.unsqueeze(0).permute(0, 3, 1, 2)}
        outputs1 = my_net._forward(inputs1, Mode.EXPORT, model_config)
        desc1 = outputs1['descriptors']

        gpu_img2 = torch.tensor(img2, dtype=torch.float, device=device)
        inputs2 = {'image0': gpu_img2.unsqueeze(0).permute(0, 3, 1, 2)}
        outputs2 = my_net._forward(inputs2, Mode.EXPORT, model_config)
        desc2 = outputs2['descriptors']

        # Sample the descriptors at the keypoint positions
        desc1 = sample_descriptors(gpu_kp1, desc1, img_size)
        desc2 = sample_descriptors(gpu_kp2, desc2, img_size)

        # Nearest neighbor matching based on the LISRD descriptors
        matches = mvnet_matcher(desc1, desc2).cpu().numpy()

    # Refine the matches with RANSAC
    matched_kp1, matched_kp2 = kp1[matches[:, 0]][:, [1, 0]], kp2[matches[:, 1]][:, [1, 0]]
    matched_kp1, matched_kp2 = filter_outliers_ransac(matched_kp1, matched_kp2)

    out = my_drawMatches(img1, matched_kp1, img2, matched_kp2)
    cv2.imwrite("test.jpg", out)
