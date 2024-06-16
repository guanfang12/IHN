import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
from skimage import io
import random
import sys
import cv2
import matplotlib.pyplot as plt 
import torchgeometry as tgm

from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
import time

def get_feature_show(feature_map):
    # 加载特征图, inputs.shape = [batch, channel,H,W]
    feature_map = feature_map.squeeze(0).permute(1,2,0)
    feature_map = feature_map.cpu().detach().numpy()

    # 将特征图展平为一维数组
    feature_vector = feature_map.reshape(-1, feature_map.shape[-1])

    # 创建PCA对象，并设置降维后的维度
    pca = PCA(n_components=3)

    # 对特征向量进行降维
    feature_reduced = pca.fit_transform(feature_vector)

    # 将降维后的特征向量重新形状为特征图的形状
    feature_reduced = feature_reduced.reshape(feature_map.shape[:-1] + (3,))

    # 将像素值限制在0到1之间
    feature_reduced = np.clip(feature_reduced, 0, 1)
    return feature_reduced

def get_BEV_tensor(img,Ho, Wo, Fov = 170, dty = -20, dx = 0, dy = 0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    t0 = time.time()
    if len(img.shape) ==3 :
        Hp, Wp , _ = img.shape                                # 全景图尺寸
    else:
        Hp, Wp  = img.shape                                # 全景图尺寸
    if dty != 0 or Wp != 2*Hp:
        ty = (Wp/2-Hp)/2 + dty                                                 # 非标准全景图补全
        matrix_K = np.array([[1,0,0],[0,1,ty],[0,0,1]])
        img = cv2.warpPerspective(img,matrix_K,(int(Wp),int(Hp+(Wp/2-Hp))))
    ######################
    t1 = time.time()
    # frame = torch.from_numpy(img.astype(np.float32)).to(device)
    frame = torch.from_numpy(img.copy()).to(device)  
    t2 = time.time()

    if len(frame.shape) ==3 :
        Hp, Wp , _ = frame.shape                                # 全景图尺寸
    else:
        Hp, Wp  = frame.shape                                # 全景图尺寸
    # Wp, Hp = 16384, 8192                                # 全景图尺寸
    Fov = Fov * torch.pi / 180                               # 视场角
    center = torch.tensor([Wp/2+dx,Hp+dy]).to(device)                  # 俯瞰图中心
    # Ho, Wo =  500,500                                        # 俯瞰图尺寸

    anglez = center[0] * 2 * torch.pi / Wp
    angley = torch.pi / 2 - center[1] * torch.pi / Hp

    f = Wo/2/torch.tan(torch.tensor(Fov/2))
    r = Rotation.from_euler('zy',[anglez.cpu(),angley.cpu()], degrees=False)
    R02 = torch.from_numpy(r.as_matrix()).float().to(device)
    out = torch.zeros((Wo, Ho,2)).to(device)
    f0 = torch.zeros((Wo, Ho,3)).to(device)  
    f0[:,:,0] = torch.ones((Wo, Ho)).to(device) *f
    f0[:,:,1] = -Wo/2 + torch.ones((Ho, Wo)).to(device)  *torch.arange(Wo).to(device)  
    f0[:,:,2] = -Ho/2 + (torch.ones((Ho, Wo)).to(device)  *(torch.arange(Ho)).to(device)).T
    f1 = R02@ f0.reshape((-1,3)).T  # x,y,z (3*N)
    f1_0 = torch.sqrt(torch.sum(f1**2,0))
    f1_1 = torch.sqrt(torch.sum(f1[:2,:]**2,0))
    theta = torch.arccos(f1[2,:]/f1_0)
    phi = torch.arccos(f1[0,:]/f1_1)
    mask = f1[1,:] <  0 
    phi[mask] = 2 * torch.pi - phi[mask]
    #################################
    phi = 2 * torch.pi - phi+ torch.pi
    mask = phi >  2 * torch.pi 
    phi[mask] = phi[mask] - 2 * torch.pi 
    #################################
    i_p = theta  / torch.pi 
    j_p = phi  / (2 * torch.pi) 
    out[:,:,0] = j_p.reshape((Ho, Wo))
    out[:,:,1] = i_p.reshape((Ho, Wo))
    t3 = time.time()
    out[:,:,0] = (out[:,:,0]-0.5)/0.5
    out[:,:,1] = (out[:,:,1]-0.5)/0.5

    BEV = F.grid_sample(frame.permute(2,0,1).unsqueeze(0).float(), out.unsqueeze(0), align_corners=True)
    t4  = time.time()
    # plt.imshow(BEV.cpu().int()[:,:,[2,1,0]])
    # plt.imshow(img[out.astype(int)[:,:,0],out.astype(int)[:,:,1]].astype(int)[:,:,[2,1,0]])
    print("Read image ues {:.2f} ms, warpPerspective image use {:.2f} ms, Get matrix ues {:.2f} ms, Get out ues {:.2f} ms, All out ues {:.2f} ms.".format((t1-t0)*1000,(t2-t1)*1000, (t3-t2)*1000,(t4-t3)*1000,(t4-t0)*1000))
    # BEV.cpu().int()
    return BEV.permute(0,2,3,1).squeeze(0).cpu().int()


def show_overlap(img1, img2, H, show = True):
    # 显示重叠, from  img1 to img2 (np.array)
    image1 = img1.copy()
    image0 = img2.copy()
    h,w = image0.shape[0],image0.shape[1]
    h_,w_ = image1.shape[0],image1.shape[1]

    result = cv2.warpPerspective(image1, H, (w+w + image1.shape[1], h)) # result 左面放置image1 warp 之后的图像
    mask_temp =result[:,0:w] > 1  
    temp2 = result[:,0:w,:].copy()
    frame = image0.astype(np.uint8)
    roi = frame[mask_temp]
    frame[mask_temp] = (0.5*temp2.astype(np.uint8)[mask_temp] + 0.5 * roi).astype(np.uint8)
    result[0:image0.shape[0],image1.shape[1]:image0.shape[1] + image1.shape[1]] = frame   # result 中间放置重叠
    result[0:h,image1.shape[1]+w : w+w + image1.shape[1]] = image0   # result 右面放置image0 
    pts = np.float32([[0, 0], [0, h_], [w_, h_], [w_, 0]]).reshape(-1, 1, 2)
    center = np.float32( [w_/2, h_/2]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    dst_center = cv2.perspectiveTransform(center, H).reshape(-1, 2)
    # 加上偏移量
    for i in range(4):
        dst[i][0] += w_+ w
    dst_center[0][0] += w_+ w
    cv2.polylines(result, [np.int32(dst)], True, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(result,(int(dst_center[0][0]),int(dst_center[0][1])),15,(0,255,0),1)
    cv2.circle(temp2,(int(dst_center[0][0]-w_- w),int(dst_center[0][1])),15,(0,255,0),1)
    if show:
        # plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.imshow(temp2.astype(np.uint8))
        plt.subplot(1,3,2)
        plt.imshow(result[:,w_:w_+w].astype(np.uint8))
        plt.subplot(1,3,3)
        plt.imshow(result[:,w+w_:].astype(np.uint8))
    return result

def get_homograpy(four_point, sz, k = 1):
    """
    four_point: four corner flow
    sz: image size
    k: scale
    Shape:
        - Input: :four_point:`(B, 2, 2, 2)` and :sz:`(B, C, H, W)`
        - Output: :math:`(B, 3, 3)`
    """
    N,_,h,w = sz
    # h, w = h//k, w//k
    four_point = four_point / k # four_point 是原图的尺寸， coordinate 是特征图尺寸
    four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
    four_point_org[:, 0, 0] = torch.Tensor([0, 0])
    four_point_org[:, 0, 1] = torch.Tensor([w-1, 0])
    four_point_org[:, 1, 0] = torch.Tensor([0, h-1])
    four_point_org[:, 1, 1] = torch.Tensor([w -1, h-1])

    four_point_org = four_point_org.unsqueeze(0)
    four_point_org = four_point_org.repeat(N, 1, 1, 1)
    four_point_new = torch.autograd.Variable(four_point_org) + four_point
    four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
    four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
    H = tgm.get_perspective_transform(four_point_org, four_point_new)
    return H

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    # [1024, 1, 32, 32]  [1024, 9, 9, 2] https://www.jb51.net/article/273930.htm
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1 # 图像尺寸归一化[-1,1]
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)  # [1024, 9, 9, 2]
    img = F.grid_sample(img, grid, align_corners=True) # [1024, 1, 9, 9]，按grid对img采样

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float() # 坐标为 (x,y)
    return coords[None].expand(batch, -1, -1, -1)

def save_img(img, path):
    npimg = img.detach().cpu().numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    npimg = npimg.astype(np.uint8)
    io.imsave(path, npimg)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(x.device)
    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output * mask

class Logger_(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

class Logger:
    def __init__(self, model, scheduler, args):
        self.model = model
        self.args = args
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss_dict = {}
        self.train_mace_list = []
        self.train_steps_list = []
        self.val_steps_list = []
        self.val_results_dict = {}

    def _print_training_status(self):
        metrics_data = [np.mean(self.running_loss_dict[k]) for k in sorted(self.running_loss_dict.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data[:-1])).format(*metrics_data[:-1])
        # Compute time left
        time_left_sec = (self.args.num_steps - (self.total_steps+1)) * metrics_data[-1]
        time_left_sec = time_left_sec.astype(np.int)
        time_left_hms = "{:02d}h{:02d}m{:02d}s".format(time_left_sec // 3600, time_left_sec % 3600 // 60, time_left_sec % 3600 % 60)
        time_left_hms = f"{time_left_hms:>12}"
        # print the training status
        print(training_str + metrics_str + time_left_hms)
        # logging running loss to total loss
        self.train_mace_list.append(np.mean(self.running_loss_dict['mace']))
        self.train_steps_list.append(self.total_steps)
        for key in self.running_loss_dict:
            self.running_loss_dict[key] = []

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss_dict:
                self.running_loss_dict[key] = []
            self.running_loss_dict[key].append(metrics[key])
        if self.total_steps % self.args.print_freq == self.args.print_freq-1:
            self._print_training_status()
            self.running_loss_dict = {}