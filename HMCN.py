'''
down_conv : 12 -->24
up : 48 -->24
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
from dropblock import DropBlock3D
dir_LFCAMorph = os.getcwd()
dir_woAttnmodels = os.path.dirname(dir_LFCAMorph)
dir_models = os.path.dirname(dir_woAttnmodels)
dir_Code = os.path.dirname(dir_models)
sys.path.append(dir_Code)
from Functions import generate_grid_unit, generate_grid


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, num_group=4, stride=1, bias=False):
        super(PreActBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias)
            )


    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=0.2)

        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)

        out = self.conv2(F.leaky_relu(out, negative_slope=0.2))


        out += shortcut
        return out


class SpatialTransform_unit(nn.Module):
    def __init__(self):
        super(SpatialTransform_unit, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='bilinear', padding_mode="border", align_corners=True)
        return flow


class SpatialTransformNearest_unit(nn.Module):
    def __init__(self):
        super(SpatialTransformNearest_unit, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='nearest', padding_mode="border", align_corners=True)
        return flow


class DiffeomorphicTransform_unit(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform_unit, self).__init__()
        self.time_step = time_step

    def forward(self, velocity, sample_grid):
        flow = velocity/(2.0**self.time_step)
        # print(f'\tflow.shape : {flow.shape}')
        # print(f'\tsample_grid.shape : {sample_grid.shape}')
        for _ in range(self.time_step):
            # a = flow.permute(0, 2, 3, 4, 1)
            # print(f'\ta.shape : {a.shape}')
            grid = sample_grid + flow.permute(0,2,3,4,1)

            # print(f'\tgrid.shape : {grid.shape}')
            flow = flow + F.grid_sample(flow, grid, mode='bilinear', padding_mode="border", align_corners=True)
        return flow


class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        size_tensor = sample_grid.size()
        sample_grid[0, :, :, :, 0] = (sample_grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (
                size_tensor[3] - 1) * 2
        sample_grid[0, :, :, :, 1] = (sample_grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (
                size_tensor[2] - 1) * 2
        sample_grid[0, :, :, :, 2] = (sample_grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (
                size_tensor[1] - 1) * 2
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='bilinear', align_corners=True)

        return flow


class SpatialTransformNearest(nn.Module):
    def __init__(self):
        super(SpatialTransformNearest, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        size_tensor = sample_grid.size()
        sample_grid[0, :, :, :, 0] = (sample_grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (
                    size_tensor[3] - 1) * 2
        sample_grid[0, :, :, :, 1] = (sample_grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (
                    size_tensor[2] - 1) * 2
        sample_grid[0, :, :, :, 2] = (sample_grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (
                    size_tensor[1] - 1) * 2
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='nearest', align_corners=True)

        return flow


class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step

    def forward(self, velocity, sample_grid, range_flow):
        flow = velocity / (2.0 ** self.time_step)
        size_tensor = sample_grid.size()
        # 0.5 flow
        for _ in range(self.time_step):
            grid = sample_grid + (flow.permute(0, 2, 3, 4, 1) * range_flow)
            grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (size_tensor[3] - 1) * 2
            grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (size_tensor[2] - 1) * 2
            grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (size_tensor[1] - 1) * 2
            flow = flow + F.grid_sample(flow, grid, mode='bilinear', align_corners=True)
        return flow


def smoothloss(y_pred):
    dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :])
    dx = torch.abs(y_pred[:,:,:, 1:, :] - y_pred[:,:, :, :-1, :])
    dz = torch.abs(y_pred[:,:,:, :, 1:] - y_pred[:,:, :, :, :-1])
    return (torch.mean(dx * dx)+torch.mean(dy*dy)+torch.mean(dz*dz))/3.0


def jacobian_det(deformation_field: torch.Tensor, lamb=None, normalize=True) -> torch.Tensor:
    # expects shape B,2,(D),H,W
    # returns shape B,1,(D),H,W")
    ndims = len(deformation_field.shape[2:])
    device = deformation_field.device

    if ndims == 2:
        if normalize:
            deformation_field = torch.stack((deformation_field[:, 0] * 2 / deformation_field.shape[-2],
                                             deformation_field[:, 1] * 2 / deformation_field.shape[-1]), 1)
        B, _, H, W = deformation_field.size()
        rep_x = nn.ReplicationPad2d((1, 1, 0, 0)).to(device)
        rep_y = nn.ReplicationPad2d((0, 0, 1, 1)).to(device)

        kernel_y = nn.Conv2d(2, 2, (3, 1), bias=False, groups=2).to(device)
        kernel_y.weight.data[:, 0, :, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(2, 1).to(device)
        kernel_x = nn.Conv2d(2, 2, (1, 3), bias=False, groups=2).to(device)
        kernel_x.weight.data[:, 0, 0, :] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(2, 1).to(device)

        disp_field_vox = deformation_field.flip(1) * (torch.Tensor([H - 1, W - 1]).to(device).view(1, 2, 1, 1) - 1) / 2
        grad_y = kernel_y(rep_y(disp_field_vox))
        grad_x = kernel_x(rep_x(disp_field_vox))

        jacobian = torch.stack((grad_y, grad_x), 1) + torch.eye(2, 2).to(device).view(1, 2, 2, 1, 1)
        jac_det = jacobian[:, 0, 0, :, :] * jacobian[:, 1, 1, :, :] - jacobian[:, 1, 0, :, :] * jacobian[:, 0, 1, :, :]
    elif ndims == 3:
        if normalize:
            deformation_field = torch.stack((deformation_field[:, 0] * 2 / deformation_field.shape[-3],
                                             deformation_field[:, 1] * 2 / deformation_field.shape[-2],
                                             deformation_field[:, 2] * 2 / deformation_field.shape[-1]), 1)

        B, _, D, H, W = deformation_field.size()
        rep_x = nn.ReplicationPad3d((1, 1, 0, 0, 0, 0)).to(device)
        rep_y = nn.ReplicationPad3d((0, 0, 1, 1, 0, 0)).to(device)
        rep_z = nn.ReplicationPad3d((0, 0, 0, 0, 1, 1)).to(device)

        kernel_z = nn.Conv3d(3, 3, (3, 1, 1), bias=False, groups=3).to(device)
        kernel_z.weight.data[:, 0, :, 0, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1).to(device)
        kernel_y = nn.Conv3d(3, 3, (1, 3, 1), bias=False, groups=3).to(device)
        kernel_y.weight.data[:, 0, 0, :, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1).to(device)
        kernel_x = nn.Conv3d(3, 3, (1, 1, 3), bias=False, groups=3).to(device)
        kernel_x.weight.data[:, 0, 0, 0, :] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1).to(device)

        disp_field_vox = deformation_field.flip(1) * (
                    torch.Tensor([D - 1, H - 1, W - 1]).to(device).view(1, 3, 1, 1, 1) - 1) / 2
        grad_z = kernel_z(rep_z(disp_field_vox))
        grad_y = kernel_y(rep_y(disp_field_vox))
        grad_x = kernel_x(rep_x(disp_field_vox))

        jacobian = torch.stack((grad_z, grad_y, grad_x), 1)
        eye = torch.eye(3, 3).to(device).view(1, 3, 3, 1, 1, 1)
        jacobian = jacobian + eye
        jac_det = (jacobian[:, 0, 0, :, :, :] *
                   (jacobian[:, 1, 1, :, :, :] * jacobian[:, 2, 2, :, :, :] - jacobian[:, 2, 1, :, :, :] * jacobian[:,1, 2, :, :,:])
                   - jacobian[:, 0,1, :,:,:] *
                   (jacobian[:, 1, 0, :, :, :] * jacobian[:, 2, 2, :, :, :] - jacobian[:, 2, 0, :, :,:] * jacobian[:, 1, 2, :, :,:])
                   + jacobian[:, 0, 2, :,:, :] *
                   (jacobian[:, 1, 0, :, :, :] * jacobian[:, 2, 1, :, :, :] - jacobian[:, 2, 0, :, :,:] * jacobian[:, 1, 1, :, :, :]))
    return jac_det


def JDetStd(deformation_field: torch.Tensor, lamb=0, normalize=True) -> torch.Tensor:
    """The standard deviation of the Jacobian determinant as a regularization loss."""
    return lamb * jacobian_det(deformation_field, normalize=normalize).std()



def JacobianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


def neg_Jdet_loss(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacobianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)


class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=5, eps=1e-8):
        super(NCC, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


class multi_resolution_NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def  __init__(self, win=None, eps=1e-5, scale=3):
        super(multi_resolution_NCC, self).__init__()
        self.num_scale = scale
        self.similarity_metric = []

        for i in range(scale):
            self.similarity_metric.append(NCC(win=win - (i*2)))

    def forward(self, I, J):
        total_NCC = []

        for i in range(self.num_scale):
            current_NCC = self.similarity_metric[i](I, J)
            total_NCC.append(current_NCC/(2**i))

            I = nn.functional.avg_pool3d(I, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            J = nn.functional.avg_pool3d(J, kernel_size=3, stride=2, padding=1, count_include_pad=False)

        return sum(total_NCC)

class kl_loss(nn.Module):
    def __init__(self):
        super(kl_loss, self).__init__()

    def __call__(self, flow_mean, flow_log_sigma):
        kl_div = -0.5 * torch.sum(1 + flow_log_sigma - flow_mean.pow(2) - flow_log_sigma.exp(), dim=[1, 2, 3, 4])
        return (kl_div.mean() / (flow_mean.size(0) * flow_mean.size(1) * flow_mean.size(2) * flow_mean.size(3) * flow_mean.size(4)))

class lamda_mse_loss(nn.Module):
    def __init__(self):
        super(lamda_mse_loss, self).__init__()
        self.image_sigma=1

    def __call__(self, x, y):
        return 1.0 / (self.image_sigma ** 2) * torch.mean( (x - y) ** 2 )


class ProbabilisticModel(nn.Module):
    def __init__(self, is_training=True):
        super(ProbabilisticModel, self).__init__()

        self.mean = torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.log_sigma = torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=3, padding=1)

        # Manual Initialization
        self.mean.weight.data.normal_(0, 1e-5)
        self.log_sigma.weight.data.normal_(0, 1e-10)
        self.log_sigma.bias.data.fill_(-10.)


        self.is_training=is_training

    def forward(self, final_layer):
        flow_mean = self.mean(final_layer)
        flow_log_sigma = self.log_sigma(final_layer)
        noise = torch.randn_like(flow_mean).cuda()

        if self.is_training:
            flow = flow_mean + flow_log_sigma * noise
        else:
            flow = flow_mean + flow_log_sigma # No noise at testing time

        return flow, flow_mean, flow_log_sigma

class contrastive_loss(nn.Module):
    def __init__(self, batch_size=8, temperature=0.5, use_cosine_similarity=True):
        super(contrastive_loss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.cuda()

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def __call__(self, zis, zjs):
        # print(1, zjs.shape)
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        # print(f'l_pos.shape : {l_pos.shape} r_pos.shape : {r_pos.shape}')
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature
        # print('logits:', logits)

        labels = torch.zeros(2 * self.batch_size).cuda().long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)



class MultiResblock_dil(torch.nn.Module):
    '''
    MultiRes Block - YG - dilated Version

    Arguments:
        num_in_channels {int} -- Number of channels coming into mutlires block
        num_filters {int} -- Number of filters in a corrsponding UNet stage
        # alpha {float} -- alpha hyperparameter (default: 1.67)

    '''

    def __init__(self, num_in_channels, num_filters, alpha=1.67):
        super().__init__()
        self.alpha = alpha
        self.W = num_filters * alpha

        if self.W <= 3:
            filt_cnt_3x3 = 1
            filt_cnt_7x7 = 1
            filt_cnt_5x5 = 1
        else:
            filt_cnt_3x3 = int(round(self.W * 0.167))
            filt_cnt_7x7 = int(round(self.W * 0.5))
            filt_cnt_5x5 = self.W - (filt_cnt_3x3 + filt_cnt_7x7)

        num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7


        self.shortcut = torch.nn.Conv3d(in_channels=num_in_channels, out_channels=num_out_filters, kernel_size=(1, 1, 1),padding=0)

        self.conv_3x3 =nn.Sequential(
            torch.nn.Conv3d(in_channels=num_in_channels, out_channels=filt_cnt_3x3, kernel_size=(3, 3, 3), padding=1),
            torch.nn.BatchNorm3d(filt_cnt_3x3)
        )

        self.conv_5x5 = nn.Sequential(
            torch.nn.Conv3d(in_channels=filt_cnt_3x3, out_channels=filt_cnt_5x5, kernel_size=(3, 3, 3), dilation=2,padding=2),
            torch.nn.BatchNorm3d(filt_cnt_5x5)
        )

        self.conv_7x7 = nn.Sequential(
            torch.nn.Conv3d(in_channels=filt_cnt_5x5, out_channels=filt_cnt_7x7, kernel_size=(3, 3, 3), dilation=3, padding=3),
            torch.nn.BatchNorm3d(filt_cnt_7x7)
        )

        self.batch_norm1 = torch.nn.BatchNorm3d(num_out_filters)
        self.batch_norm2 = torch.nn.BatchNorm3d(num_out_filters)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        shrtct = self.shortcut(x)
        # print(" MRD shrtct.shape : {}".format(shrtct.shape))
        a = self.conv_3x3(x)
        # print(" MRD a.shape : {}".format(a.shape))
        b = self.conv_5x5(a)
        # print(" MRD b.shape : {}".format(b.shape))
        c = self.conv_7x7(b)
        # print(" MRD c.shape : {}".format(c.shape))

        x = torch.cat([a, b, c], axis=1)
        # print(" MRD x.shape : {}".format(x.shape))
        x = self.batch_norm1(x)
        # print(" MRD x.shape after batch_norm1: {}".format(x.shape))

        x = x + shrtct
        # print(" MRD x.shape after + :  {}".format(x.shape))
        x = self.batch_norm2(x)
        # print(" MRD x.shape after batch_norm2: {}".format(x.shape))
        # x = torch.nn.functional.relu(x)
        x = self.relu(x)
        # print(" MRD x.shape : {}".format(x.shape))
        # print()
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, batchnorm = True):
        '''
        Block for convolutional layer of U-Net at the encoder end.
        Args:
            ch_in : number of input channels
            ch_out : number of outut channels
        Returns:
            feature map of the given input
        '''
        super(conv_block, self).__init__()
        if batchnorm :
            self.conv = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
                nn.BatchNorm3d(ch_out),
                nn.ReLU(inplace=True),
                nn.Conv3d(ch_out, ch_out, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
                nn.BatchNorm3d(ch_out),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(ch_out, ch_out, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
                nn.ReLU(inplace=True)
            )
    def forward(self, x):
        x = self.conv(x)
        return x

class MultiAttention(nn.Module):
    def __init__(self, F_g, F_l, F_int, n_groups=4,):
        super(MultiAttention, self).__init__()

        self.W_g_01 = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=True),
            nn.Conv3d(F_int, F_int, kernel_size=(3, 3, 3), dilation=1, padding=1),
            nn.BatchNorm3d(F_int))
        # nn.GroupNorm(n_groups, F_int))
        self.W_x_01 = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=True),
            nn.Conv3d(F_int, F_int, kernel_size=(3, 3, 3), dilation=2, padding=2),
            nn.BatchNorm3d(F_int))
        # nn.GroupNorm(n_groups, F_int))

        self.W_g_02 = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=True),
            nn.Conv3d(F_int, F_int, kernel_size=(3, 3, 3), dilation=5, padding=5),
            nn.BatchNorm3d(F_int))
        # nn.GroupNorm(n_groups, F_int))
        self.W_x_02 = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=True),
            nn.Conv3d(F_int, F_int, kernel_size=(3, 3, 3), dilation=7, padding=7),
            nn.BatchNorm3d(F_int))
        # nn.GroupNorm(n_groups, F_int))

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True),
            # nn.GroupNorm(n_groups, 1),
            nn.BatchNorm3d(1),
            nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g_01(g)
        x1 = self.W_x_01(x)
        c1 = self.relu(g1 + x1)
        g2 = self.W_g_02(g)
        x2 = self.W_x_02(x)
        c2 = self.relu(g2 + x2)
        c_comb = self.relu(c1 + c2)
        psi_01 = self.psi(c_comb)
        return x * psi_01


class model_diff_l1(nn.Module):
    def __init__(self, in_channel, ch_start, ch_start_magnitude, n_classes, is_train=True, imgshape=(160, 192, 144), range_flow=0.4,doubleF=True, doubleB=True, batchsize=1):
        super(model_diff_l1, self).__init__()

        self.in_channel = in_channel
        self.ch_start = ch_start
        self.ch_start_magnitude = ch_start_magnitude
        self.n_classes = n_classes
        self.factor = 3
        self.batchsize = batchsize
        self.doubleF = doubleF
        self.doubleB = doubleB
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear_x = torch.nn.Linear(self.ch_start * ch_start_magnitude, self.ch_start * ch_start_magnitude)
        self.linear_y = torch.nn.Linear(self.ch_start * ch_start_magnitude, self.ch_start * ch_start_magnitude)
        self.range_flow = range_flow
        self.is_train = is_train
        self.imgshape = imgshape
        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()
        self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().cuda()


        bias_opt = False

        self.input_encoder_lvl1 = self.multiresblock_dil_front(self.in_channel//2, self.ch_start * ch_start_magnitude,
                                                               doubleF=self.doubleF)
        self.down_conv = nn.Conv3d(self.ch_start * self.ch_start_magnitude, self.ch_start * self.ch_start_magnitude, 3, stride=2, padding=1,
                                   bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.ch_start * ch_start_magnitude*2, bias_opt=bias_opt)

        self.up = nn.ConvTranspose3d(self.ch_start * ch_start_magnitude*2 , self.ch_start * ch_start_magnitude, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.attention = MultiAttention(self.ch_start * self.ch_start_magnitude, self.ch_start * self.ch_start_magnitude*2,
                                                       self.ch_start * self.ch_start_magnitude*2)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.multiresblock_dil_back(self.ch_start * ch_start_magnitude*self.factor, self.n_classes, doubleB=self.doubleB)

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        )
        return layer

    def multiresblock_dil_front(self, in_channels, out_channels, doubleF=False):
        # print("multiresblock_dil_front in_channels : {}, out_channels : {}".format(in_channels, out_channels))
        # print(f"doubleF : {doubleF}")
        if doubleF:
            layer = nn.Sequential(
                MultiResblock_dil(in_channels, int(out_channels / 2), 1),
                MultiResblock_dil(int(out_channels / 2), out_channels, 1),
                nn.Softsign()
            )
        else:
            layer = nn.Sequential(
                MultiResblock_dil(in_channels, out_channels, 1)
            )
        # print(f"layer : {layer}")
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def multiresblock_dil_back(self, in_channels, out_channels, doubleB=False):
        # print("multiresblock_dil_back in_channels : {}, out_channels : {}".format(in_channels,out_channels))
        # print(f"doubleB : {doubleB}")
        if doubleB:
            layer = nn.Sequential(
                MultiResblock_dil(in_channels, int(in_channels / 2), 1),
                MultiResblock_dil(int(in_channels / 2), out_channels, 1),
                nn.Softsign())
        else:
            layer = nn.Sequential(
                MultiResblock_dil(in_channels, out_channels, 1))

        return layer

    def forward(self, x, y):
        x_down = self.down_avg(x)
        y_down = self.down_avg(y)

        x_down2 = self.down_avg(x_down)
        y_down2 = self.down_avg(y_down)

        fea_e0_x = self.input_encoder_lvl1(x_down2)
        f_x = self.avgpool(fea_e0_x)
        f_x = f_x.squeeze()

        if self.batchsize == 1:
            f_x = f_x.unsqueeze(dim=0)

        f_x = self.linear_x(f_x)
        f_x = f_x / f_x.norm(dim=-1, keepdim=True)

        fea_e0_y = self.input_encoder_lvl1(y_down2)
        f_y = self.avgpool(fea_e0_y)
        f_y = f_y.squeeze()

        if self.batchsize == 1:
            f_y = f_y.unsqueeze(dim=0)

        f_y = self.linear_x(f_y)
        f_y = f_y / f_y.norm(dim=-1, keepdim=True)
        fea_e0 = torch.cat((fea_e0_x, fea_e0_y), 1)

        e0_x = self.down_conv(fea_e0_x)
        e0_y = self.down_conv(fea_e0_y)
        e0 = torch.cat((e0_x, e0_y), 1)
        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)
        attn = self.attention(e0, fea_e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([attn, e0], dim=1)) * self.range_flow

        output_disp_e0 = self.diff_transform(output_disp_e0_v, self.grid_1)
        warpped_inputx_lvl1_out = self.transform(x, output_disp_e0.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return output_disp_e0, warpped_inputx_lvl1_out, y_down2, output_disp_e0_v, e0, f_x, f_y
        else:
            return output_disp_e0



class model_diff_l2(nn.Module):
    def __init__(self, in_channel, ch_start, ch_start_magnitude, n_classes, is_train=True, imgshape=(160, 192, 144), range_flow=0.4,doubleF=True, doubleB=True,
                 model_lvl1=None, batchsize=1):
        super(model_diff_l2, self).__init__()

        self.in_channel = in_channel
        self.n_classes = n_classes
        self.ch_start = ch_start
        self.ch_start_magnitude = ch_start_magnitude
        self.factor = 3
        self.doubleF = doubleF
        self.doubleB = doubleB

        self.batchsize = batchsize
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear_x = torch.nn.Linear(self.ch_start * self.ch_start_magnitude, self.ch_start * self.ch_start_magnitude)
        self.linear_y = torch.nn.Linear(self.ch_start * self.ch_start_magnitude, self.ch_start * self.ch_start_magnitude)
        self.range_flow = range_flow
        self.is_train = is_train
        self.imgshape = imgshape
        self.model_lvl1 = model_lvl1
        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().cuda()

        bias_opt = False
        self.input_encoder_lvl1 = self.multiresblock_dil_front(self.in_channel//2+3, self.ch_start * self.ch_start_magnitude,
                                                               doubleF=self.doubleF)
        self.down_conv = nn.Conv3d(self.ch_start * ch_start_magnitude, self.ch_start * ch_start_magnitude, 3, stride=2, padding=1, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.ch_start * ch_start_magnitude*2, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)#align_corners=False
        self.up = nn.ConvTranspose3d(self.ch_start * ch_start_magnitude*2, self.ch_start * ch_start_magnitude, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)
        self.attention = MultiAttention(self.ch_start * self.ch_start_magnitude, self.ch_start * self.ch_start_magnitude*2,
                                                       self.ch_start * self.ch_start_magnitude*2)
        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        self.output_lvl1 = self.multiresblock_dil_back(self.ch_start * ch_start_magnitude*self.factor, self.n_classes, doubleB=self.doubleB)


    def unfreeze_modellvl1(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl1 parameter")
        for param in self.model_lvl1.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        )
        return layer



    def multiresblock_dil_front(self, in_channels, out_channels, doubleF=False):
        # print("multiresblock_dil_front in_channels : {}, out_channels : {}".format(in_channels, out_channels))
        # print(f"doubleF : {doubleF}")
        if doubleF:
            layer = nn.Sequential(
                MultiResblock_dil(in_channels, int(out_channels / 2), 1),
                MultiResblock_dil(int(out_channels / 2), out_channels, 1),
                nn.Softsign()
            )
        else:
            layer = nn.Sequential(
                MultiResblock_dil(in_channels, out_channels, 1)
            )
        # print(f"layer : {layer}")
        return layer


    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer


    def multiresblock_dil_back(self, in_channels, out_channels, doubleB=False):
        # print("multiresblock_dil_back in_channels : {}, out_channels : {}".format(in_channels,out_channels))
        # print(f"doubleB : {doubleB}")
        if doubleB:
            layer = nn.Sequential(
                MultiResblock_dil(in_channels, int(in_channels / 2), 1),
                # nn.LeakyReLU(0.2),
                MultiResblock_dil(int(in_channels / 2), out_channels, 1),
                nn.Softsign()
            )
        else:
            layer = nn.Sequential(
                MultiResblock_dil(in_channels, out_channels, 1)
            )
        return layer



    def forward(self, x, y):
        lvl1_disp, _, _, lvl1_v, lvl1_embedding, f_x_l1, f_x_l2 = self.model_lvl1(x, y)
        lvl1_disp_up = self.up_tri(lvl1_disp)
        lvl1_v_up = self.up_tri(lvl1_v)
        x_down = self.down_avg(x)
        y_down = self.down_avg(y)
        warpped_x = self.transform(x_down, lvl1_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)
        cat_input_lvl2_x = torch.cat((warpped_x, lvl1_v_up), 1)
        fea_e0_x = self.input_encoder_lvl1(cat_input_lvl2_x)
        f_x = self.avgpool(fea_e0_x)
        f_x = f_x.squeeze()
        if self.batchsize == 1:
            f_x = f_x.unsqueeze(dim=0)
        f_x = self.linear_x(f_x)
        f_x = f_x / f_x.norm(dim=-1, keepdim=True)
        cat_input_lvl2_y = torch.cat((y_down, lvl1_v_up), 1)
        fea_e0_y = self.input_encoder_lvl1(cat_input_lvl2_y)

        f_y = self.avgpool(fea_e0_y)
        f_y = f_y.squeeze()
        if self.batchsize == 1:
            f_y = f_y.unsqueeze(dim=0)
        f_y = self.linear_y(f_y)
        f_y = f_y / f_y.norm(dim=-1, keepdim=True)

        e0_x = self.down_conv(fea_e0_x)
        e0_y = self.down_conv(fea_e0_y)
        e0_x = e0_x + lvl1_embedding
        e0_y = e0_y + lvl1_embedding
        e0 = torch.cat((e0_x, e0_y),1)
        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)

        fea_e0 = torch.cat((fea_e0_x, fea_e0_y),1)
        attn = self.attention(e0, fea_e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([attn, e0], dim=1)) * self.range_flow

        compose_field_e0_lvl1v = output_disp_e0_v + lvl1_v_up
        output_disp_e0 = self.diff_transform(compose_field_e0_lvl1v, self.grid_1)
        warpped_inputx_lvl1_out = self.transform(x, output_disp_e0.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return output_disp_e0, warpped_inputx_lvl1_out, y_down, compose_field_e0_lvl1v, lvl1_v, e0, f_x, f_y
        else:
            return output_disp_e0




class model_diff_l3(nn.Module):
    def __init__(self, in_channel, ch_start, ch_start_magnitude, n_classes, is_train=True, imgshape=(160, 192, 144), range_flow=0.4,doubleF=True, doubleB=True,
                 model_lvl2=None, batchsize=1):
        super(model_diff_l3, self).__init__()

        self.in_channel = in_channel
        self.n_classes = n_classes
        self.ch_start = ch_start
        self.ch_start_magnitude = ch_start_magnitude
        self.factor = 3
        self.batchsize = batchsize
        self.doubleF = doubleF
        self.doubleB = doubleB

        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear_x = torch.nn.Linear(self.ch_start * self.ch_start_magnitude,
                                        self.ch_start * self.ch_start_magnitude)
        self.linear_y = torch.nn.Linear(self.ch_start * self.ch_start_magnitude,
                                        self.ch_start * self.ch_start_magnitude)
        self.range_flow = range_flow
        self.is_train = is_train
        self.imgshape = imgshape
        self.model_lvl2 = model_lvl2
        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()
        self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().cuda()

        bias_opt = False
        self.input_encoder_lvl1 = self.multiresblock_dil_front(self.in_channel // 2 + 3,
                                                               self.ch_start * self.ch_start_magnitude,
                                                               doubleF=self.doubleF)
        self.down_conv = nn.Conv3d(self.ch_start * ch_start_magnitude, self.ch_start * ch_start_magnitude, 3, stride=2, padding=1,
                                   bias=bias_opt)
        self.resblock_group_lvl1 = self.resblock_seq(self.ch_start * ch_start_magnitude*2, bias_opt=bias_opt)
        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.up = nn.ConvTranspose3d(self.ch_start * ch_start_magnitude*2, self.ch_start * ch_start_magnitude, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.attention = MultiAttention(self.ch_start * self.ch_start_magnitude,
                                            self.ch_start * self.ch_start_magnitude * 2,
                                            self.ch_start * self.ch_start_magnitude * 2)

        self.output_lvl1 = self.multiresblock_dil_back(self.ch_start * ch_start_magnitude * self.factor, self.n_classes,
                                                       doubleB=self.doubleB)

    def unfreeze_modellvl2(self):
        # unFreeze model_lvl1 weight
        # print("\nunfreeze model_lvl2 parameter")
        for param in self.model_lvl2.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        )
        return layer


    def multiresblock_dil_front(self, in_channels, out_channels, doubleF=False):
        # print("multiresblock_dil_front in_channels : {}, out_channels : {}".format(in_channels, out_channels))
        # print(f"doubleF : {doubleF}")
        if doubleF:
            layer = nn.Sequential(
                MultiResblock_dil(in_channels, int(out_channels / 2), 1),
                MultiResblock_dil(int(out_channels / 2), out_channels, 1),
                nn.Softsign()
            )
        else:
            layer = nn.Sequential(
                MultiResblock_dil(in_channels, out_channels, 1)
            )
        # print(f"layer : {layer}")
        return layer




    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def multiresblock_dil_back(self, in_channels, out_channels, doubleB=False):
        # print("multiresblock_dil_back in_channels : {}, out_channels : {}".format(in_channels, out_channels))
        # print(f"doubleB : {doubleB}")
        if doubleB:
            layer = nn.Sequential(
                MultiResblock_dil(in_channels, int(in_channels / 2), 1),
                # nn.LeakyReLU(0.2),
                MultiResblock_dil(int(in_channels / 2), out_channels, 1),
                nn.Softsign()
            )
        else:
            layer = nn.Sequential(
                MultiResblock_dil(in_channels, out_channels, 1)
            )
        # print(f"layer : {layer}")
        return layer


    def forward(self, x, y):
        lvl2_disp, _, _,compose_lvl2_v, lvl1_v, lvl2_embedding, f_x_l2, f_y_l2 = self.model_lvl2(x, y)
        lvl2_disp_up = self.up_tri(lvl2_disp)
        compose_lvl2_v_up = self.up_tri(compose_lvl2_v)
        warpped_x = self.transform(x, lvl2_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)
        cat_input_x = torch.cat((warpped_x, compose_lvl2_v_up), 1)
        fea_e0_x = self.input_encoder_lvl1(cat_input_x)
        f_x = self.avgpool(fea_e0_x)
        f_x = f_x.squeeze()
        if self.batchsize == 1:
            f_x = f_x.unsqueeze(dim=0)
        f_x = self.linear_x(f_x)
        f_x = f_x / f_x.norm(dim=-1, keepdim=True)

        cat_input_y = torch.cat((y, compose_lvl2_v_up), 1)
        fea_e0_y = self.input_encoder_lvl1(cat_input_y)
        f_y = self.avgpool(fea_e0_y)
        f_y = f_y.squeeze()
        if self.batchsize == 1:
            f_y = f_y.unsqueeze(dim=0)

        f_y = self.linear_y(f_y)
        f_y = f_y / f_y.norm(dim=-1, keepdim=True)

        e0_x = self.down_conv(fea_e0_x)
        e0_y = self.down_conv(fea_e0_y)

        e0_x = e0_x + lvl2_embedding
        e0_y = e0_y + lvl2_embedding

        e0 = torch.cat((e0_x, e0_y), 1)
        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)

        fea_e0 = torch.cat((fea_e0_x, fea_e0_y), 1)
        attn = self.attention(e0, fea_e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([attn, e0], dim=1)) * self.range_flow

        compose_field_e0_lvl2_compose = output_disp_e0_v + compose_lvl2_v_up
        output_disp_e0 = self.diff_transform(compose_field_e0_lvl2_compose, self.grid_1)
        warpped_inputx_lvl1_out = self.transform(x, output_disp_e0.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return output_disp_e0, warpped_inputx_lvl1_out, y, compose_field_e0_lvl2_compose, lvl1_v, compose_lvl2_v, e0, f_x, f_y
        else:
            return output_disp_e0


if __name__ == "__main__":
    imgshape = (144, 160, 192)
    imgshape_4 = (144 // 4, 160 // 4, 192 // 4)
    imgshape_2 = (144 // 2, 160 // 2, 192 // 2)
    ch_start = 6
    ch_start_magnitude = 2
    range_flow = 0.4

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input1 = torch.randn(1, 1, 144, 160, 192).cuda()
    input2 = torch.randn(1, 1, 144, 160, 192).cuda()

    X = input1.to(device).float()
    Y = input1.to(device).float()


    model_names = ['MRD_diff']
    attns = ['multi_attn']
    concats = ['e0']
    doubleF = True
    doubleB = True



    for model_name in model_names:
        print(f'model_name : {model_name}')
        for attn in attns:
            print(f'attn : {attn}')
            for concat in concats:
                model_lvl1 = model_diff_l1(2, ch_start, ch_start_magnitude, 3, is_train=True, imgshape=imgshape_4,
                                           range_flow=range_flow, doubleF=doubleF, doubleB=doubleB, batchsize=1).to(device)
                model_lvl2 = model_diff_l2(2, ch_start, ch_start_magnitude, 3, is_train=True, imgshape=imgshape_2,
                                           range_flow=range_flow, model_lvl1=model_lvl1, doubleF=doubleF, doubleB=doubleB, batchsize=1).to(device)

                model = model_diff_l3(2,ch_start, ch_start_magnitude, 3, is_train=True, imgshape=imgshape,
                                      range_flow=range_flow, model_lvl2=model_lvl2, doubleF=doubleF,doubleB=doubleB, batchsize=1).to(device)
                F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1,_,_ , f_x, f_y = model(X, Y)
                print(f'F_X_Y.shape : {F_X_Y.shape}')
