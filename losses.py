""" 
Depth Loss by Alhashim et al.:

Ibraheem Alhashim, High Quality Monocular Depth Estimation via
Transfer Learning, https://arxiv.org/abs/1812.11941, 2018

https://github.com/ialhashim/DenseDepth
"""

import torch
import torch.nn.functional as F

from math import exp

class Depth_Loss():
    def __init__(self, alpha, beta, gamma, maxDepth=10.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.maxDepth = maxDepth

        self.L1_Loss = torch.nn.L1Loss()


    def __call__(self, output, depth, features):
        if self.beta == 0 and self.gamma == 0:
            valid_mask = depth>0.0
            output = output[valid_mask]
            depth = depth[valid_mask]
            l_depth = self.L1_Loss(output, depth)
            loss = l_depth
        else:
            l_depth = self.L1_Loss(output, depth)
            l_ssim = torch.clamp((1-self.ssim(output, depth, self.maxDepth)) * 0.5, 0, 1)
            l_grad = self.gradient_loss(output, depth)

            loss = self.alpha * l_depth + self.beta * l_ssim + self.gamma * l_grad
        #return loss

            oe_loss = self.ordinalentropy(features, depth)        # new lines for ordinal
            loss = loss + oe_loss        # new lines for ordinal
        return loss        # new lines for ordinal


    def ssim(self, img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
        L = val_range

        padd = 0
        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = self.create_window(real_size, channel=channel).to(img1.device)
            padd = window_size // 2

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs

        return ret


    def gradient_loss(self, gen_frames, gt_frames, alpha=1):
        gen_dx, gen_dy = self.gradient(gen_frames)
        gt_dx, gt_dy = self.gradient(gt_frames)

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        # condense into one tensor and avg
        grad_comb = grad_diff_x ** alpha + grad_diff_y ** alpha

        return torch.mean(grad_comb)


    def gradient(self, x):
        """
        idea from tf.image.image_gradients(image)
        https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        """
        h_x = x.size()[-2]
        w_x = x.size()[-1]

        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        dx, dy = right - left, bottom - top

        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy


    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window


    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum() 

    def ordinalentropy(self, features, gt,  mask=None):
        """
        Features: a certain layer's features
        gt: pixel-wise ground truth values, in depth estimation, gt.size()= n, h, w
        mask: In case values of some pixels do not exist. For depth estimation, there are some pixels lack the ground truth values
        """
        f_n, f_c, f_h, f_w = features.size()

        features = F.interpolate(features, size=[f_h // 4, f_w // 4], mode='nearest')
        features = features.permute(0, 2, 3, 1)  # n, h, w, c
        features = torch.flatten(features, start_dim=1, end_dim=2)

        gt = F.interpolate(gt, size=[f_h // 4, f_w // 4], mode='nearest')

        loss = 0

        for i in range(f_n):
            """
            mask pixels that without valid values
            """
            _gt = gt[i,:].view(-1)
            _mask = _gt > 0.001
            _mask = _mask.to(torch.bool)
            _gt = _gt[_mask]
            _features = features[i,:]
            _features = _features[_mask,:]

            """
            diverse part
            """
            u_value, u_index, u_counts = torch.unique(_gt, return_inverse=True, return_counts =True)
            center_f = torch.zeros([len(u_value), f_c]).cuda()
            center_f.index_add_(0, u_index, _features)
            u_counts = u_counts.unsqueeze(1)
            center_f = center_f / u_counts

            p = F.normalize(center_f, dim=1)
            _distance = self.euclidean_dist(p, p)
            _distance = self.up_triu(_distance)

            u_value = u_value.unsqueeze(1)
            _weight = self.euclidean_dist(u_value, u_value)
            _weight = self.up_triu(_weight)
            _max = torch.max(_weight)
            _min = torch.min(_weight)
            _weight = ((_weight - _min) / _max)
    
            _distance = _distance * _weight

            _entropy = torch.mean(_distance)
            loss = loss - _entropy

            """
            tightness part
            
            _features = F.normalize(_features, dim=1)
            _features_center = p[u_index, :]
            _features = _features - _features_center
            _features = _features.pow(2)
            _tightness = torch.sum(_features, dim=1)
            _mask = _tightness > 0
            _tightness = _tightness[_mask]

            _tightness = torch.sqrt(_tightness)
            _tightness = torch.mean(_tightness)

            loss = loss + _tightness
            """
        return loss/ f_n

    def euclidean_dist(self, x, y):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist


    def up_triu(self, x):
        # return a flattened view of up triangular elements of a square matrix
        n, m = x.shape
        assert n == m
        _tmp = torch.triu(torch.ones(n, n), diagonal=1).to(torch.bool)
        return x[_tmp]
