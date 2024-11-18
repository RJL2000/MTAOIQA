import torch
import torch.nn.functional as F
import timm
import torch.nn as nn
import itertools
from einops import rearrange
import math
import numpy as np
from scipy import signal
from config_mta import config_mta

def creat_model(config, pretrained=False):
    model_mta_oiqa = MTA_OIQA(embed_dim=config.embed_dim)
    if pretrained:
        model_mta_oiqa.load_state_dict(torch.load(config.model_weight_path), strict=False)
        print("weights had been load!\n")
    return model_mta_oiqa


def globale_mean_std_pool2d(x):
    mean = nn.functional.adaptive_avg_pool2d(x, 1)
    std = global_std_pool2d(x)
    result = torch.cat([mean, std], 1)
    return result


def global_avg_pool2d(x):
    mean = nn.functional.adaptive_avg_pool2d(x, 1)
    return mean


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),dim=2, keepdim=True)


# 特征图大小 （128，28，28）
class Multitask_FeatureSelect(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(256, embed_dim, 1, 1, 0)
        self.conv2 = nn.Conv2d(512, embed_dim, 1, 1, 0)
        self.conv3 = nn.Conv2d(1024, embed_dim, 1, 1, 0)
        self.conv4 = nn.Conv2d(1024, embed_dim, 1, 1, 0)
        self.conv5 = nn.Conv2d(384, embed_dim, 1, 1, 0)
        self.conv2_1 = nn.Conv2d(embed_dim, embed_dim, 4, 4, 0)
        self.conv3_1 = nn.Conv2d(embed_dim, embed_dim, 2, 2, 0)
        self.fc1 = nn.Linear(128*8,15, bias=False)
        self.fc2 = nn.Linear(128*8,4, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, xs):

        ## swintransformer
        x1, x2, x3, x4 = xs[0], xs[1], xs[2], xs[3]
        x1 = rearrange(x1, 'V_n (h w) c -> V_n c h w', h=28, w=28)
        x2 = rearrange(x2, 'V_n (h w) c -> V_n c h w', h=14, w=14)
        x3 = rearrange(x3, 'V_n (h w) c -> V_n c h w', h=7, w=7)
        x4 = rearrange(x4, 'V_n (h w) c -> V_n c h w', h=7, w=7)

        x1 = self.conv1(x1) # torch.Size([8, 128, 28, 28]) 
        x2 = self.conv2(x2) # torch.Size([8, 128, 14, 14]) 
        x3 = self.conv3(x3) # torch.Size([8, 128, 7, 7])
        x4 = self.conv4(x4) # torch.Size([8, 128, 7, 7])

        x2_1 = F.interpolate(x2, size=28, mode='bilinear', align_corners=False)
        x3_1 = F.interpolate(x3, size=28, mode='bilinear', align_corners=False)
        x4_1 = F.interpolate(x4, size=28, mode='bilinear', align_corners=False)

        x1_2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear', align_corners=False)
        x3_2 = F.interpolate(x3, size=14, mode='bilinear', align_corners=False)
        x4_2 = F.interpolate(x4, size=14, mode='bilinear', align_corners=False)

        x1_3 = F.interpolate(x1, scale_factor=0.25, mode='bilinear', align_corners=False)
        x2_3 = F.interpolate(x2, scale_factor=0.5, mode='bilinear', align_corners=False)

        x1_4 = F.interpolate(x1, scale_factor=0.25, mode='bilinear', align_corners=False)
        x2_4 = F.interpolate(x2, scale_factor=0.5, mode='bilinear', align_corners=False)

        x1 = x1 + x2_1 + x3_1 + x4_1 # torch.Size([8, 128, 28, 28]) 
        x2 = x1_2 + x2 + x3_2 + x4_2 # torch.Size([8, 128, 14, 14]) 
        x3 = x1_3 + x2_3 + x3 + x4 # torch.Size([8, 128, 7, 7])
        x4 = x1_4 + x2_4 + x3 + x4 # torch.Size([8, 128, 7, 7])

        x2 = F.interpolate(x2, size=28, mode='bilinear', align_corners=False) # torch.Size([8, 128, 28, 28])
        x3 = F.interpolate(x3, size=28, mode='bilinear', align_corners=False) # torch.Size([8, 128, 28, 28])
        x4 = F.interpolate(x4, size=28, mode='bilinear', align_corners=False) # torch.Size([8, 128, 28, 28])

        s_feat = global_avg_pool2d(self.conv2(torch.cat((x1, x2, x3, x4), dim=1))).flatten(0)  # )# torch.Size([8*128])
        # print(s_feat.shape)

        v_feat1, v_feat2 = self.fc1(s_feat), self.fc2(s_feat)  # torch.Size([15]), torch.Size([4])

        v_feat2 = self.softmax(v_feat2)
        # print(v_feat2)

        combinations = []
        for r in range(1, 5):
            combinations.extend(itertools.combinations([x1 * v_feat2[0], x2 * v_feat2[1], x3 * v_feat2[2], x4 * v_feat2[3]], r))

        index = torch.argmax(self.softmax(v_feat1), dim=0)  # torch.Size([1])
        
    

        x = combinations[index]

        if len(x) == 1:
            x = x[0]
        elif len(x) == 2:
            x = self.conv1(torch.cat((x[0], x[1]), dim=1))
        elif len(x) == 3:
            x = self.conv5(torch.cat((x[0], x[1], x[2]), dim=1))
        else:
            x = self.conv2(torch.cat((x[0], x[1], x[2], x[3]), dim=1))

        return x


class Viewport_Distortion(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(embed_dim, embed_dim // 4, 1, 1, 0)
        self.conv1 = nn.Conv2d(embed_dim // 4, 1, 1, 1, 0)
        self.conv_c = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.conv_c1 = nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)
        self.conv3 = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        x1 = self.gap(x)  #  (8, 128, 1, 1)
        x1 = self.conv(x1) #  （8, 64, 1, 1）
        x1 = self.gelu(x1)
        x1 = self.conv1(x1) #  （8, 1, 1, 1）
        x1 = self.sigmoid(x1)

        x1 = x1 * x # (8, 128, 7, 7)
        x3 = self.conv3(torch.cat((x, x1), dim=1))

        return x3


class Spatial_Distortion(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(embed_dim, embed_dim // 4, 1, 1, 0)
        self.conv1 = nn.Conv2d(embed_dim // 4, 1, 1, 1, 0)
        self.conv_c = nn.Conv2d(embed_dim, 1, 1, 1, 0)
        self.conv_c1 = nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)
        self.conv3 = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        self.dilated_conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=3, dilation=3)
        self.dilated_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=10, dilation=5)
        self.dilated_conv3 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1, dilation=1)
        self.conv_c2 = nn.Conv2d(64 + 128 + 1, 1, 1, 1, 0)


    def forward(self, x):

        x1 = self.gap(x)  #  (8, 128, 1, 1)
        x1 = self.conv(x1) #  （8, 64, 1, 1）
        x1 = self.gelu(x1)
        x1 = self.conv1(x1) #  （8, 1, 1, 1）
        x1 = self.sigmoid(x1)

        x1 = x1 * x # (8, 128, 7, 7)

        x2 = self.conv_c(x)  # (8, 1, 7, 7)

        x2_1 = self.dilated_conv1(x2)
        x2_2 = self.dilated_conv2(x2_1)
        x2_3 = self.dilated_conv3(x2_2)

        x2 = self.conv_c2(torch.cat((x2_1, x2_2, x2_3), dim=1))

        x2 = self.sigmoid(x2)  # (8, 1, 7, 7)

        x2 = x2 * x

        x3 = self.conv3(torch.cat((x1, x2), dim=1))

        return x3
    

class Channel_Distortion(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(embed_dim, embed_dim // 4, 1, 1, 0)
        self.conv1 = nn.Conv2d(embed_dim // 4, 1, 1, 1, 0)
        self.conv_c = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.conv_c1 = nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)
        self.conv3 = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        x1 = self.gap(x)  #  (8, 128, 1, 1)
        x1 = self.conv(x1) #  （8, 64, 1, 1）
        x1 = self.gelu(x1)
        x1 = self.conv1(x1) #  （8, 1, 1, 1）
        x1 = self.sigmoid(x1)

        x1 = x1 * x # (8, 128, 7, 7)

        x2 = self.conv_c(x)  # (8, 64, 7, 7)
        x2 = self.gelu(x2)
        x2 = self.conv_c1(x2)  # (8, 128, 7, 7)
        x2 = self.sigmoid(x2)  # (8, 128, 7, 7)

        x3 = self.conv3(torch.cat((x1, x2), dim=1))

        return x3


class Multidimension_Distortion(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(embed_dim, embed_dim // 4, 1, 1, 0)
        self.conv1 = nn.Conv2d(embed_dim // 4, 1, 1, 1, 0)
        self.conv_c = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.conv_c1 = nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)
        self.conv_c2 = nn.Conv2d(embed_dim, 1, 1, 1, 0)

        self.conv3 = nn.Conv2d(embed_dim * 3, embed_dim, 1, 1, 0)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        self.dilated_conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=3, dilation=3)
        self.dilated_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=10, dilation=5)
        self.dilated_conv3 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1, dilation=1)
        self.conv_c3 = nn.Conv2d(64+128+1, 1, 1, 1, 0)


    def forward(self, x):

        x1 = self.gap(x)  #  (8, 128, 1, 1)
        x1 = self.conv(x1) #  （8, 64, 1, 1）
        x1 = self.gelu(x1)
        x1 = self.conv1(x1) #  （8, 1, 1, 1）
        x1 = self.sigmoid(x1)

        x1 = x1 * x # (8, 128, 7, 7)

        x2 = self.conv_c(x)  # (8, 64, 7, 7)
        x2 = self.gelu(x2)
        x2 = self.conv_c1(x2)  # (8, 128, 7, 7)
        x2 = self.sigmoid(x2)  # (8, 128, 7, 7)

        x3 = self.conv_c2(x)
        x3_1 = self.dilated_conv1(x3)
        x3_2 = self.dilated_conv2(x3_1)
        x3_3 = self.dilated_conv3(x3_2)

        x3 = self.conv_c3(torch.cat((x3_1, x3_2, x3_3), dim=1))

        x3 = self.sigmoid(x3)
        x3 = x * x3

        x4 = self.conv3(torch.cat((x1, x2, x3), dim=1))

        return x4
    

class Spatial_Distortion_Perception(nn.Module):
    def __init__(self, embed_dim=128, vac_layers=3):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        vca = []
        for i in range(vac_layers):
            vca += [Spatial_Distortion(embed_dim=embed_dim)]
        self.mdp = nn.Sequential(*vca)

    def forward(self, x):

        x = self.mdp(x)
        x = self.gap(x)

        return x


class Viewport_Distortion_Perception(nn.Module):
    def __init__(self, embed_dim=128, vac_layers=3):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        vca = []
        for i in range(vac_layers):
            vca += [Viewport_Distortion(embed_dim=embed_dim)]
        self.mdp = nn.Sequential(*vca)

    def forward(self, x):

        x = self.mdp(x)
        x = self.gap(x)

        return x
    

class Channel_Distortion_Perception(nn.Module):
    def __init__(self, embed_dim=128, vac_layers=3):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        vca = []
        for i in range(vac_layers):
            vca += [Channel_Distortion(embed_dim=embed_dim)]
        self.mdp = nn.Sequential(*vca)

    def forward(self, x):

        x = self.mdp(x)
        x = self.gap(x)

        return x
    

class Multidimension_Distortion_Perception(nn.Module):
    def __init__(self, embed_dim=128, vac_layers=3):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        vca = []
        for i in range(vac_layers):
            vca += [Multidimension_Distortion(embed_dim=embed_dim)]
        self.mdp = nn.Sequential(*vca)

    def forward(self, x):

        x = self.mdp(x)
        x = self.gap(x)

        return x


class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)

        return output


class Multiauxiliary_Task_Fusion(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.liner_q = nn.Linear(embed_dim * 8, embed_dim, bias=None)
        self.liner_t = nn.Linear(config_mta().type_num, embed_dim, bias=None)
        self.liner_r = nn.Linear(config_mta().range_num, embed_dim, bias=None)
        self.liner_d = nn.Linear(config_mta().degree_num, embed_dim, bias=None)
        self.liner_ta4 = nn.Linear(embed_dim * 4, embed_dim, bias=None)
        self.liner_ta3 = nn.Linear(embed_dim * 3, embed_dim, bias=None)
        self.liner_ta2 = nn.Linear(embed_dim * 2, embed_dim, bias=None)
        self.ca = CrossAttention(in_dim1=128, in_dim2=128, k_dim=256, v_dim=256, num_heads=4)

        self.liner = nn.Linear(embed_dim * 3, embed_dim, bias=None)
    
    def forward(self, **kwargs):

        x_q = kwargs.get('quality')
        x_r = kwargs.get('range')
        x_t = kwargs.get('type')
        x_d = kwargs.get('degree')

        x_q = x_q.permute(2, 3, 0, 1).flatten(1) # 1, 128*8*1
        x_q = self.liner_q(x_q).unsqueeze(0) # 1 , 1, 128

        if x_r is not None:
            x_r = self.liner_r(x_r).unsqueeze(0)   # (1, 2)->(1,1,128)
            
        
        if x_t is not None:
            x_t = self.liner_t(x_t).unsqueeze(0)   # (1, 4)->(1,1,128)
        
        if x_d is not None:
            x_d = self.liner_d(x_d).unsqueeze(0)   # (1, 3)->(1,1,128)
        
        if x_r is not None and x_t is not None and x_d is not None:
            x_q_r_t_d = torch.cat((x_q, x_r, x_t, x_d), dim=2)
            x_q_r_t_d = self.liner_ta4(x_q_r_t_d)

            x_ca1 = self.ca(x_q_r_t_d, x_q)
            x_ca2 = self.ca(x_q, x_q_r_t_d)
            x_ca3 = self.ca(x_q_r_t_d, x_q_r_t_d)

            x_ca = self.liner(torch.cat((x_ca1, x_ca2, x_ca3), dim=2).flatten(1))  #(1,128)

            return x_ca

        elif x_r is not None and x_t is not None:
            x_q_r_t = torch.cat((x_q, x_r, x_t), dim=2)
            x_q_r_t = self.liner_ta3(x_q_r_t)

            x_ca1 = self.ca(x_q_r_t, x_q)
            x_ca2 = self.ca(x_q, x_q_r_t)
            x_ca3 = self.ca(x_q_r_t, x_q_r_t)

            x_ca = self.liner(torch.cat((x_ca1, x_ca2, x_ca3), dim=2).flatten(1))  #(1,128)

            return x_ca
        
        elif x_r is not None and x_d is not None:
            x_q_r_d = torch.cat((x_q, x_r, x_d), dim=2)
            x_q_r_d = self.liner_ta3(x_q_r_d)

            x_ca1 = self.ca(x_q_r_d, x_q)
            x_ca2 = self.ca(x_q, x_q_r_d)
            x_ca3 = self.ca(x_q_r_d, x_q_r_d)

            x_ca = self.liner(torch.cat((x_ca1, x_ca2, x_ca3), dim=2).flatten(1))  #(1,128)

            return x_ca
        
        elif x_t is not None and x_d is not None:
            x_q_t_d = torch.cat((x_q, x_t, x_d), dim=2)
            x_q_t_d = self.liner_ta3(x_q_t_d)

            x_ca1 = self.ca(x_q_t_d, x_q)
            x_ca2 = self.ca(x_q, x_q_t_d)
            x_ca3 = self.ca(x_q_t_d, x_q_t_d)

            x_ca = self.liner(torch.cat((x_ca1, x_ca2, x_ca3), dim=2).flatten(1))  #(1,128)

            return x_ca
        
        elif x_t is not None:
            x_q_t = torch.cat((x_q, x_t), dim=2)
            x_q_t = self.liner_ta2(x_q_t)

            x_ca1 = self.ca(x_q_t, x_q)
            x_ca2 = self.ca(x_q, x_q_t)
            x_ca3 = self.ca(x_q_t, x_q_t)

            x_ca = self.liner(torch.cat((x_ca1, x_ca2, x_ca3), dim=2).flatten(1))  #(1,128)

            return x_ca
        
        elif x_r is not None:
            x_q_r = torch.cat((x_q, x_r), dim=2)
            x_q_r = self.liner_ta2(x_q_r)

            x_ca1 = self.ca(x_q_r, x_q)
            x_ca2 = self.ca(x_q, x_q_r)
            x_ca3 = self.ca(x_q_r, x_q_r)

            x_ca = self.liner(torch.cat((x_ca1, x_ca2, x_ca3), dim=2).flatten(1))  #(1,128)

            return x_ca
        
        elif x_d is not None:
            x_q_d = torch.cat((x_q, x_d), dim=2)
            x_q_d = self.liner_ta2(x_q_d)

            x_ca1 = self.ca(x_q_d, x_q)
            x_ca2 = self.ca(x_q, x_q_d)
            x_ca3 = self.ca(x_q_d, x_q_d)

            x_ca = self.liner(torch.cat((x_ca1, x_ca2, x_ca3), dim=2).flatten(1))  #(1,128)

            return x_ca
        else:
    
            x_ca1 = self.ca(x_q, x_q)
            x_ca2 = self.ca(x_q, x_q)
            x_ca3 = self.ca(x_q, x_q)

            x_ca = self.liner(torch.cat((x_ca1, x_ca2, x_ca3), dim=2).flatten(1))  #(1,128)

            return x_ca


class Distortion_Type_pred(nn.Module):
    def __init__(self, embed_dim=128, num_classes=config_mta().type_num):
        super().__init__()
        
        self.sdp = Spatial_Distortion_Perception(embed_dim=embed_dim, vac_layers=4)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)
        self.liner1 = nn.Linear(embed_dim*8, embed_dim, bias=False)
        self.liner2 = nn.Linear(embed_dim, num_classes, bias=False)
    def forward(self, x):

        # x_type = self.channel(x)
        x_type = self.sdp(x)

        # print(x_tpye.shape)
        x_t = x_type.flatten(2) # 8, 128 ,1 
        x_t = x_t.permute(2, 0, 1).flatten(1) # 1 ,128*8
        # print(x_t.shape)

        x_t = self.liner1(x_t)
        x_t = self.gelu(x_t)
        x_t = self.liner2(x_t)
        pred_type = self.softmax(x_t) # (1, 3)
        pred_type_ = pred_type.flatten(0) #(3)

        return pred_type, pred_type_
    

class Distortion_Range_pred(nn.Module):
    def __init__(self, embed_dim=128, num_classes=config_mta().range_num):
        super().__init__()
        
        self.vdp = Viewport_Distortion_Perception(embed_dim=embed_dim, vac_layers=8)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)
        self.liner1 = nn.Linear(embed_dim*8, embed_dim, bias=False)
        self.liner2 = nn.Linear(embed_dim, num_classes, bias=False)

    
    def forward(self, x):

        # x_range = self.channel(x)
        x_range = self.vdp(x)


        x_r = x_range.flatten(2) # 8, 128 ,1 ,1
        x_r = x_r.permute(2, 0, 1).flatten(1) # 1 ,128*8
        # print(x_t.shape)

        x_r = self.liner1(x_r)
        x_r = self.gelu(x_r)
        x_r = self.liner2(x_r)
        pred_range = self.softmax(x_r) # (1, 2)
        pred_range_ = pred_range.flatten(0) #(2)


        return pred_range, pred_range_
    

class Distortion_Degree_pred(nn.Module):
    def __init__(self, embed_dim=128, num_classes=config_mta().degree_num):
        super().__init__()
        
        self.cdp = Channel_Distortion_Perception(embed_dim=embed_dim, vac_layers=4)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)
        self.liner1 = nn.Linear(embed_dim*8, embed_dim, bias=False)
        self.liner2 = nn.Linear(embed_dim, num_classes, bias=False)

    def forward(self, x):
        
        x_degree = self.cdp(x)
        x_d = x_degree.flatten(2) # 8, 128 ,1 
        x_d = x_d.permute(2, 0, 1).flatten(1) # 1 ,128*8
        # print(x_t.shape)

        x_d = self.liner1(x_d)
        x_d = self.gelu(x_d)
        x_d = self.liner2(x_d)
        pred_degree = self.softmax(x_d) # (1, 3)
        pred_degree_ = pred_degree.flatten(0) #(3)

        return pred_degree, pred_degree_


class Distortion_Quality_pred(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        
        self.mdp = Multidimension_Distortion_Perception(embed_dim=embed_dim, vac_layers=4)
        self.conv = nn.Conv2d(embed_dim*4, embed_dim, 1, 1, 0)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)
        self.liner1 = nn.Linear(embed_dim, embed_dim // 2, bias=False)
        self.liner2 = nn.Linear(embed_dim // 2, 1, bias=False)
        self.liner3 = nn.Linear(8, 1, bias=False)
        self.mtf = Multiauxiliary_Task_Fusion()



    def forward(self, **kwargs):

        x_q = kwargs.get('quality')
        x_r = kwargs.get('range')
        x_t = kwargs.get('type')
        x_d = kwargs.get('degree')

        if x_r is not None and x_t is not None and x_d is not None:
            x_q = self.mdp(x_q)
            x_q = self.mtf(quality=x_q, type=x_t, range=x_r, degree=x_d)
            x_q = self.liner1(x_q)  # 1 ,128
            x_q = self.gelu(x_q)
            x_q = self.liner2(x_q)  # 1 ,1
            x_quality = x_q.flatten(0) # 1
            
            return x_quality

        elif x_r is not None and x_t is not None:
            x_q = self.mdp(x_q)
            x_q = self.mtf(quality=x_q, type=x_t, range=x_r)
            x_q = self.liner1(x_q)  # 1 ,128
            x_q = self.gelu(x_q)
            x_q = self.liner2(x_q)  # 1 ,1
            x_quality = x_q.flatten(0) # 1 

            return x_quality
        elif x_r is not None and x_d is not None:
            x_q = self.mdp(x_q)
            x_q = self.mtf(quality=x_q, range=x_r, degree=x_d)
            x_q = self.liner1(x_q)  # 1 ,128
            x_q = self.gelu(x_q)
            x_q = self.liner2(x_q)  # 1 ,1
            x_quality = x_q.flatten(0) # 1 

            return x_quality
        elif x_t is not None and x_d is not None:
            x_q = self.mdp(x_q)
            x_q = self.mtf(quality=x_q, type=x_t, degree=x_d)
            x_q = self.liner1(x_q)  # 1 ,128
            x_q = self.gelu(x_q)
            x_q = self.liner2(x_q)  # 1 ,1
            x_quality = x_q.flatten(0) # 1 
            return x_quality
        elif x_r is not None:
            x_q = self.mdp(x_q)
            x_q = self.mtf(quality=x_q, range=x_r)
            x_q = self.liner1(x_q)  # 1 ,128
            x_q = self.gelu(x_q)
            x_q = self.liner2(x_q)  # 1 ,1
            x_quality = x_q.flatten(0) # 1 

            return x_quality
        elif x_t is not None:
            x_q = self.mdp(x_q)
            x_q = self.mtf(quality=x_q, type=x_t)
            x_q = self.liner1(x_q)  # 1 ,128
            x_q = self.gelu(x_q)
            x_q = self.liner2(x_q)  # 1 ,1
            x_quality = x_q.flatten(0) # 1 

            return x_quality
        elif x_d is not None:
            x_q = self.mdp(x_q)
            x_q = self.mtf(quality=x_q, degree=x_d)
            x_q = self.liner1(x_q)  # 1 ,128
            x_q = self.gelu(x_q)
            x_q = self.liner2(x_q)  # 1 ,1
            x_quality = x_q.flatten(0) # 1 

            return x_quality
        else:
            x_q = self.mdp(x_q)
            x_q = self.mtf(quality=x_q)
            x_q = self.liner1(x_q)  # 8 ,128
            x_q = self.gelu(x_q)
            x_q = self.liner2(x_q)  # 1 ,1
            
            x_quality = x_q.flatten(0) # 1 
            return x_quality

 
class MTA_OIQA(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.act_layer = nn.GELU()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', checkpoint_path='/mnt/10T/rjl/swin_base_patch4_window7_224_22kto1k.pth')
        self.conv1_1 = nn.Conv2d(256, embed_dim, 1, 1, 0)
        self.conv2_1 = nn.Conv2d(512, embed_dim, 1, 1, 0)
        self.conv3_1 = nn.Conv2d(1024, embed_dim, 1, 1, 0)
        self.conv4_1 = nn.Conv2d(1128, embed_dim, 1, 1, 0)
        self.liner1 = nn.Linear(128, 1, bias=False)

        self.select = Multitask_FeatureSelect()

        self.dtp = Distortion_Type_pred()
        self.drp = Distortion_Range_pred()
        self.ddp = Distortion_Degree_pred()
        self.dqp = Distortion_Quality_pred()
    

    def forward(self, x):
        B, V, C, H, W = x.shape  # V: 视口的数量

        qualitys = torch.tensor([]).cuda()
        types = torch.tensor([]).cuda()
        ranges = torch.tensor([]).cuda()
        degrees = torch.tensor([]).cuda()


        if config_mta().qrtd:
            for i in range(B):
                pred_quality, pred_type, pred_range, pred_degree = self.vp_forward_qrtd(x[i]) #torch.Size([8, 128])
                qualitys = torch.cat((qualitys, pred_quality.unsqueeze(0)), dim=0)
                types = torch.cat((types, pred_type.unsqueeze(0)), dim=0)
                ranges = torch.cat((ranges, pred_range.unsqueeze(0)), dim=0)
                degrees = torch.cat((degrees, pred_degree.unsqueeze(0)), dim=0)

            x_q = qualitys
            x_t = types
            x_r = ranges
            x_d = degrees

            return x_q, x_t, x_r, x_d
        elif config_mta().qrt:
            for i in range(B):
                pred_quality, pred_type, pred_range = self.vp_forward_qrt(x[i]) #torch.Size([8, 128])
                qualitys = torch.cat((qualitys, pred_quality.unsqueeze(0)), dim=0)
                types = torch.cat((types, pred_type.unsqueeze(0)), dim=0)
                ranges = torch.cat((ranges, pred_range.unsqueeze(0)), dim=0)

            x_q = qualitys
            x_t = types
            x_r = ranges

            return x_q, x_t, x_r   
        elif config_mta().qrd:
            for i in range(B):
                pred_quality, pred_range, pred_degree = self.vp_forward_qrd(x[i]) #torch.Size([8, 128])
                qualitys = torch.cat((qualitys, pred_quality.unsqueeze(0)), dim=0)
                ranges = torch.cat((ranges, pred_range.unsqueeze(0)), dim=0)
                degrees = torch.cat((degrees, pred_degree.unsqueeze(0)), dim=0)

            x_q = qualitys
            x_r = ranges
            x_d = degrees

            return x_q, x_r, x_d     
        elif config_mta().qtd:
            for i in range(B):
                pred_quality, pred_type, pred_degree = self.vp_forward_qtd(x[i]) #torch.Size([8, 128])
                qualitys = torch.cat((qualitys, pred_quality.unsqueeze(0)), dim=0)
                types = torch.cat((types, pred_type.unsqueeze(0)), dim=0)
                degrees = torch.cat((degrees, pred_degree.unsqueeze(0)), dim=0)
            x_q = qualitys
            x_t = types
            x_d = degrees
            return x_q, x_t, x_d
        elif config_mta().qr:
            for i in range(B):
                pred_quality, pred_range = self.vp_forward_qr(x[i]) #torch.Size([8, 128])
                qualitys = torch.cat((qualitys, pred_quality.unsqueeze(0)), dim=0)
                ranges = torch.cat((ranges, pred_range.unsqueeze(0)), dim=0)
            x_q = qualitys
            x_r = ranges

            return x_q, x_r
        elif config_mta().qt:
            for i in range(B):
                pred_quality, pred_type = self.vp_forward_qt(x[i]) #torch.Size([8, 128])
                qualitys = torch.cat((qualitys, pred_quality.unsqueeze(0)), dim=0)
                types = torch.cat((types, pred_type.unsqueeze(0)), dim=0)
            x_q = qualitys
            x_t = types

            return x_q, x_t
        elif config_mta().qd:
            for i in range(B):
                pred_quality, pred_degree = self.vp_forward_qd(x[i]) #torch.Size([8, 128])
                qualitys = torch.cat((qualitys, pred_quality.unsqueeze(0)), dim=0)
                degrees = torch.cat((degrees, pred_degree.unsqueeze(0)), dim=0)
            x_q = qualitys
            x_d = degrees

            return x_q, x_d
        else:
            for i in range(B):
                pred_quality = self.vp_forward_q(x[i]) #torch.Size([8, 128])
                qualitys = torch.cat((qualitys, pred_quality.unsqueeze(0)), dim=0)
            x_q = qualitys
            return x_q

            


    def vp_forward_qrtd(self, x):

        x3, x0, x1, x2 = self.backbone(x)
       
        xs = [x0, x1, x2, x3]

        x_feat1 =self.select(xs)#torch.Size([8, 128, 7, 7])
        x_feat2, r, r_i =self.select(xs)#torch.Size([8, 128, 7, 7])
        x_feat3, d, d_i =self.select(xs)#torch.Size([8, 128, 7, 7])
        x_feat4, q, q_i =self.select(xs)#torch.Size([8, 128, 7, 7])
        

        
        pred_type, pred_type_ = self.dtp(x_feat1)
        
        pred_range, pred_range_ = self.drp(x_feat2)
        
        pred_degree, pred_degree_ = self.ddp(x_feat3)
        
        x_q = self.dqp(quality=x_feat4, type=pred_type, range=pred_range, degree=pred_degree)

        return x_q, pred_type_, pred_range_, pred_degree_
    
    def vp_forward_qtd(self, x):

        x3, x0, x1, x2 = self.backbone(x)
       
        xs = [x0, x1, x2, x3]

        x_feat1 =self.select(xs)#torch.Size([8, 128, 7, 7])
        x_feat3 =self.select(xs)#torch.Size([8, 128, 7, 7])
        x_feat4 =self.select(xs)#torch.Size([8, 128, 7, 7])
        
        pred_type, pred_type_ = self.dtp(x_feat1)
        
        pred_degree, pred_degree_ = self.ddp(x_feat3)
        
        x_q = self.dqp(quality=x_feat4, type=pred_type, degree=pred_degree)

        return x_q, pred_type_, pred_degree_
    
    def vp_forward_qrd(self, x):

        x3, x0, x1, x2 = self.backbone(x)
       
        xs = [x0, x1, x2, x3]

        x_feat2 =self.select(xs)#torch.Size([8, 128, 7, 7])
        x_feat3 =self.select(xs)#torch.Size([8, 128, 7, 7])
        x_feat4 =self.select(xs)#torch.Size([8, 128, 7, 7])
        
        pred_range, pred_range_ = self.drp(x_feat2)
        
        pred_degree, pred_degree_ = self.ddp(x_feat3)
        
        x_q = self.dqp(quality=x_feat4, range=pred_range, degree=pred_degree)

        return x_q, pred_range_, pred_degree_
    
    def vp_forward_qrt(self, x):

        x3, x0, x1, x2 = self.backbone(x)
       
        xs = [x0, x1, x2, x3]

        x_feat1 =self.select(xs)#torch.Size([8, 128, 7, 7])
        x_feat2 =self.select(xs)#torch.Size([8, 128, 7, 7])
        x_feat4 =self.select(xs)#torch.Size([8, 128, 7, 7])
        
        pred_type, pred_type_ = self.dtp(x_feat1)
        
        pred_range, pred_range_ = self.drp(x_feat2)
    
        
        x_q = self.dqp(quality=x_feat4, type=pred_type, range=pred_range)

        return x_q, pred_type_, pred_range_
    
    def vp_forward_qr(self, x):

        x3, x0, x1, x2 = self.backbone(x)
       
        xs = [x0, x1, x2, x3]

        x_feat2 =self.select(xs)#torch.Size([8, 128, 7, 7])
        x_feat4 =self.select(xs)#torch.Size([8, 128, 7, 7])
        
        pred_range, pred_range_ = self.drp(x_feat2)
    
        
        x_q = self.dqp(quality=x_feat4, range=pred_range)

        return x_q, pred_range_
    
    def vp_forward_qt(self, x):

        x3, x0, x1, x2 = self.backbone(x)
       
        xs = [x0, x1, x2, x3]

        x_feat1 =self.select(xs)#torch.Size([8, 128, 7, 7])
        x_feat4 =self.select(xs)#torch.Size([8, 128, 7, 7])
        
        pred_type, pred_type_ = self.dtp(x_feat1)
    
        
        x_q = self.dqp(quality=x_feat4, type=pred_type)

        return x_q, pred_type_
    
    def vp_forward_qd(self, x):

        x3, x0, x1, x2 = self.backbone(x)
        xs = [x0, x1, x2, x3]

        x_feat3 =self.select(xs)#torch.Size([8, 128, 7, 7])
        x_feat4 =self.select(xs)#torch.Size([8, 128, 7, 7])
        
        pred_degree, pred_degree_ = self.ddp(x_feat3)
        
        x_q = self.dqp(quality=x_feat4, degree=pred_degree)

        return x_q, pred_degree_
    
    def vp_forward_q(self, x):

        x3, x0, x1, x2 = self.backbone(x)
        xs = [x0, x1, x2, x3]
        x_feat4 =self.select(xs)#torch.Size([8, 128, 7, 7])
        x_q = self.dqp(quality=x_feat4)
        return x_q