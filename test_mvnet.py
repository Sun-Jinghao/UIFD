import cv2
import numpy as np
import torch
import torch.nn.functional as func
from models.DesNet.Hr30_forward import HrNet
from models.keypoint_detectors import SP_detect, load_SP_net
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
        cv2.circle(out, (int(np.round(x1)), int(np.round(y1))), 2, (255, 0, 0), 2)
        cv2.circle(out, (int(np.round(x2) + cols1), int(np.round(y2))), 2, (255, 0, 0), 2)
        cv2.line(out, (int(np.round(x1)), int(np.round(y1))), (int(np.round(x2) + cols1), int(np.round(y2))), (255, 0, 0), 1,cv2.LINE_AA)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out


def _adapt_weight_names(state_dict):
    """ Adapt the weight names when the training and testing are done
    with a different GPU configuration (with/without DataParallel). """
    train_parallel = list(state_dict.keys())[0][:7] == 'module.'
    test_parallel = torch.cuda.device_count() > 1
    new_state_dict = {}
    if train_parallel and (not test_parallel):
        # Need to remove 'module.' from all the variable names
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
    elif test_parallel and (not train_parallel):
        # Need to add 'module.' to all the variable names
        for k, v in state_dict.items():
            new_k = 'module.' + k
            new_state_dict[new_k] = v
    else:  # Nothing to do
        new_state_dict = state_dict
    return new_state_dict

def _match_state_dict(old_state_dict, new_state_dict):
    """ Return a new state dict that has exactly the same entries
            as old_state_dict and that is updated with the values of
            new_state_dict whose entries are shared with old_state_dict.
            This allows loading a pre-trained network. """
    return ({k: new_state_dict[k] if k in new_state_dict else v
             for (k, v) in old_state_dict.items()},
            old_state_dict.keys() == new_state_dict.keys())

if __name__ == "__main__":
    # load images
    img1 = cv2.cvtColor(cv2.imread(''), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(''), cv2.COLOR_BGR2RGB)

    #img1 = cv2.cvtColor(cv2.imread('/home/ouc/zh/LISRD-master/assets/03s/4/1.jpg'), cv2.COLOR_BGR2RGB)
    #img2 = cv2.cvtColor(cv2.imread('/home/ouc/zh/LISRD-master/assets/03s/4/5.jpg'),
    #img2 = cv2.cvtColor(cv2.imread('/home/ouc/zh/LISRD-master/assets/03s/4/5.jpg'), cv2.COLOR_BGR2RGB)

    img_size_ori = img1.shape[:2]  # height width
    # resize images
    #dim = (320, 240)  # width height
    #img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_CUBIC)
    #img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_CUBIC)
    img_size = img1.shape[:2]  # height width

    # Define Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the DeLoF-VoTer model
    checkpoint_path = ''
    mvnet = HrNet(128, 3)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    adapt_dict = _adapt_weight_names(checkpoint['model_state_dict'])
    net_dict = mvnet.state_dict()
    updated_state_dict, same_net = _match_state_dict(net_dict, adapt_dict)
    mvnet.load_state_dict(updated_state_dict)
    mvnet = mvnet.to(device)
    if same_net:
        print("Success in loading model!")
    mvnet.eval()

    # Load the keypoint model, here SuperPoint #内部指定了路径载入model
    kp_net = load_SP_net(conf_thresh=0.015, cuda=torch.cuda.is_available(), nms_dist=4, nn_thresh=0.7)#nms_dist=4,0.7

    with torch.no_grad():
        # Keypoint detection
        kp1 = SP_detect(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), kp_net)
        gpu_kp1 = torch.tensor(kp1, dtype=torch.float, device=device)[:, :2]
        kp2 = SP_detect(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), kp_net)
        gpu_kp2 = torch.tensor(kp2, dtype=torch.float, device=device)[:, :2]

        # Descriptor inference #opencv h w c
        gpu_img1 = torch.tensor(img1, dtype=torch.float, device=device)
        inputs1 = gpu_img1.unsqueeze(0).permute(0, 3, 1, 2)
        outputs1 = mvnet.forward(inputs1)
        desc1 ={}
        desc1['descriptors'] = outputs1 #字典

        gpu_img2 = torch.tensor(img2, dtype=torch.float, device=device)
        inputs2 = gpu_img2.unsqueeze(0).permute(0, 3, 1, 2)
        outputs2 = mvnet.forward(inputs2)
        desc2 = {}
        desc2['descriptors'] = outputs2 #字典

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
