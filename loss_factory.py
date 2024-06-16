import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils  import get_homograpy
import torchgeometry as tgm

def sequence_loss(four_pred, flow_gt, gamma=0.8):
    """ Loss function defined over sequence of coners flow predictions """
    n_predictions = len(four_pred)    
    flow_loss = 0.0

    flow_4cor = torch.zeros((four_pred[0].shape[0], 2, 2, 2)).cuda()
    flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
    flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
    flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
    flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        
        i_loss = (four_pred[i] - flow_4cor).abs()
        flow_loss += i_weight * i_loss.nanmean()

    epe = torch.sum((four_pred[-1] - flow_4cor)**2, dim=1).sqrt()

    metrics = { # 表示指标，用于给人看效果如何
        'epe': epe.nanmean().item(), # 所有角点的平均估计误差
        '1px': (epe < 1).float().mean().item(), # 表示像素误差小于1的所占百分比
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def l1(x, y, mask=None, mean = True):
    """
    Pixelwise reconstruction error Or Mean reconstruction error
    Args:
        x: predicted image
        y: target image
        mask: compute only on these points
    """
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.float32)
    if mean:
        return torch.nansum(mask * torch.abs(x - y)) / torch.sum(mask)
    return mask * torch.abs(x - y)

def SSIM(x, y, mean = True):
    """
    SSIM dissimilarity measure
    Args:
        x: predicted image
        y: target image
        mean: Mean error over SSIM reconstruction
    """

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=0)
    mu_y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=0)

    sigma_x = F.avg_pool2d(x ** 2, kernel_size=3, stride=1, padding=0) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, kernel_size=3, stride=1, padding=0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, kernel_size=3, stride=1, padding=0) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    if mean:
        return torch.nanmean(torch.clamp((1 - SSIM) / 2, 0, 1)) # https://blog.csdn.net/u013230189/article/details/82627375
    else:
        return torch.clamp((1 - SSIM) / 2, 0, 1)

def ssim_l1(x,y,alpha=0.85):
    # https://blog.csdn.net/qq_41731861/article/details/120777887
    ss = F.pad(SSIM(x, y, False), (1, 1, 1, 1))
    ll = F.l1_loss(x, y)
    return alpha * ss + (1 - alpha) * ll

def mean_SSIM_L1(x, y):
    # ssim_l, ssim_mask = ssim_loss(x, y,window_size=7)    # mean_SSIM(x, y)
    ssim_l = SSIM(x, y)
    if torch.isnan(ssim_l):
        print("ssim_l is nan!")
    l1_l = 0.5*l1(x, y)
    if torch.isnan(l1_l):
        print("l1_l is nan!")
    return 0.85* ssim_l + 0.15 * l1_l

def ssim_loss(x, y, c1=0.01**2, c2=0.03**2, window_size=3, size_average=True):
    # x和y分别为两个特征图，c1和c2为常数，window_size为滑动窗口的大小
    window = create_window(window_size, x.shape[1]).to(x.device).float()
    # https://blog.csdn.net/qq_39861441/article/details/120285815
    mu_x = F.conv2d(x, window, padding=window_size//2, groups=x.shape[1])
    mu_y = F.conv2d(y, window, padding=window_size//2, groups=y.shape[1])
    sigma_x = F.conv2d(x*x, window, padding=window_size//2, groups=x.shape[1]) - mu_x**2
    sigma_y = F.conv2d(y*y, window, padding=window_size//2, groups=y.shape[1]) - mu_y**2
    sigma_xy = F.conv2d(x*y, window, padding=window_size//2, groups=x.shape[1]) - mu_x*mu_y

    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d

    # return 1 - ssim.mean() if size_average else 1 - ssim.mean(1).mean(1).mean(1) , ssim
    ssim_l =  torch.clamp((1 - ssim) / 2, 0, 1)
    mask = (x>0).to(x.device)
    ssim = mask*ssim
    ssim_l = ssim_l.sum()/mask.sum() # + (x>=0).to(x.device).sum()/mask.sum()
    return ssim_l, ssim

def create_window(window_size, channel):
    # 创建一个高斯窗口
    window = torch.tensor([np.exp(-np.square(x - window_size//2) / (2 * np.square(1.5))) for x in range(window_size)])
    window_2d = window.unsqueeze(0).unsqueeze(1) * window.unsqueeze(1).unsqueeze(0)
    window_2d = window_2d.expand(channel, 1, window_size, window_size).contiguous() # C_out, C_in, H, W
    return window_2d / window_2d.sum()

def  neg_log_loss(corr, k = [5,24*24]):
    """
    corr: [batch, channel, h ,w]
    """
    N,C,ht,wd = corr.shape
    x = corr.reshape(N,C,ht*wd) # [batch,C, ht*wd]
    y1 = x[:,:C//2,:]
    y2 = x[:,-C//2:,:]
    softmax_y1 = torch.softmax(y1, dim=1)
    max_values_y1, max_indices = torch.topk(softmax_y1, 1, dim=1) # [batch, k ,ht*wd]
    softmax_y2 = torch.softmax(y2, dim=1)
    max_values_y2, max_indices = torch.topk(softmax_y2, 1, dim=1) # [batch, k ,ht*wd]
    # max_values_sum = torch.sum(max_values, dim=1, keepdim=True)  # [batch, 1 ,ht*wd]
    # max_values, max_indices = torch.topk(max_values, k[1], dim=2)
    # n_loss = (-torch.log(max_values)).mean()
    return (-torch.log(max_values_y1)).mean()+(-torch.log(max_values_y2)).mean()

def vigor_loss(x, y):
    epe = torch.sum((x - y)**2, dim=1).sqrt()*0.101*640/128

    metrics = { # 表示指标，用于给人看效果如何
        'epe': epe.nanmean().item(), # 所有角点的平均估计误差
        '1px': (epe < 1).float().mean().item(), # 表示真实距离误差小于1的所占百分比
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    return torch.nanmean((x-y)**2), metrics

def sim_loss(x,y,four_pred, gamma = 0.85):
    """ Loss function defined over sequence of  img similarity """
    n_predictions = len(four_pred)    
    sim_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        H = get_homograpy(four_pred[i],x.shape)
        warped = tgm.warp_perspective(y, torch.inverse(H), x.shape[-2:])
        i_loss = mean_SSIM_L1(warped, x)   
        sim_loss += i_weight * i_loss

    return sim_loss, warped