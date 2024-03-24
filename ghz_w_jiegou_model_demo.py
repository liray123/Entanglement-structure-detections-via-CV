import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class fc(nn.Module):
    def __init__(self,lizi, hid1,hid2,hid3,hid4,ylen):
        super(fc, self).__init__()
        # self.DGCNN = DGCNN_cls(inp_dim, k, emb_dims,dropout,  kernel, output_channels=2)
        self.fc1 = nn.Linear(2**(2*lizi),hid1,bias=True)
        self.fc2 = nn.Linear(hid1, hid2,bias=True)
        self.fc3 = nn.Linear(hid2, hid3, bias=True)
        self.fc4 = nn.Linear(hid3, hid4, bias=True)
        self.fc5 = nn.Linear(hid4,ylen)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x1 = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x1))
        out = self.softmax(self.fc5(self.relu(self.fc4(self.relu(self.fc3(x))))))
        return out.squeeze()


class cnn_fc(nn.Module):
    def __init__(self,lizi, hid1,hid2,hid3,hid4,ylen):
        super(cnn_fc, self).__init__()
        # self.DGCNN = DGCNN_cls(inp_dim, k, emb_dims,dropout,  kernel, output_channels=2)
        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,padding=1,bias=False,stride=2)
        self.cnn2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1,bias=False,stride=2)
        # self.cnn3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, padding=3,stride=2)
        # self.cnn4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1,stride=1)
        self.maxpoll1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        # self.fc1 = nn.Linear(2 ** (2 * lizi), hid1, bias=True)
        self.fc1 = nn.Linear(2**(2*lizi)+2**(2*lizi-2)+2**(2*lizi-4),hid1,bias=True)
        self.fc2 = nn.Linear(hid1, hid2,bias=True)
        self.fc3 = nn.Linear(hid2, hid3, bias=True)
        # self.fc4 = nn.Linear(hid3, hid4, bias=True)
        self.fc5 = nn.Linear(hid2,ylen)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.cnn1(x)
        x1 = x1.max(dim=1, keepdim=False)[0]
        x1 = x1.unsqueeze(1)
        x2 = self.cnn1(x1)
        x2 = x2.max(dim=1, keepdim=False)[0]
        x2 = x2.unsqueeze(1)
        [B, C, H, W] = x.size()
        x = x.view(B, H * W)
        x1 = x1.view(B, int((H * W) / 4))
        x2 = x2.view(B, int((H * W) / 16))
        x = torch.cat((x, x1, x2), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.softmax(self.fc5(x))
        return out.squeeze()

class cnn_vit_fc(nn.Module):
    def __init__(self,lizi, hid1,hid2,hid3,hid4,ylen,emb):
        super(cnn_vit_fc, self).__init__()
        # self.DGCNN = DGCNN_cls(inp_dim, k, emb_dims,dropout,  kernel, output_channels=2)
        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,padding=1,bias=False,stride=2)
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1,bias=False,stride=2)
        # self.cnn3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, padding=3,stride=2)
        # self.cnn4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1,stride=1)
        self.vit1 = ViT(image_channels=8, image_size=2**(lizi-1), num_classes=emb, patch_size=8, dim=192, num_heads=6,
               layers=1)
        self.vit2 = ViT(image_channels=8, image_size=2**(lizi-2), num_classes=emb, patch_size=8, dim=192, num_heads=6,
               layers=1)
        # self.fc1 = nn.Linear(2 ** (2 * lizi), hid1, bias=True)
        self.fc1 = nn.Linear(2**(2*lizi)+2*emb,hid1,bias=True)
        self.fc2 = nn.Linear(hid1, hid2,bias=True)
        self.fc3 = nn.Linear(hid2, hid3, bias=True)
        # self.fc4 = nn.Linear(hid3, hid4, bias=True)
        self.fc5 = nn.Linear(hid2,ylen)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.cnn1(x)
        x11 = self.vit1(x1)
        x2 = self.cnn2(x1)
        x22 = self.vit2(x2)
        [B, C, H, W] = x.size()
        x = x.view(B, H * W)
        x = torch.cat((x, x11, x22), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.softmax(self.fc5(x))
        return out.squeeze()



class Embedding(nn.Module):  # Patch Embedding + Position Embedding + Class Embedding
    def __init__(self, image_channels=16, image_size=224, patch_size=16, dim=768, drop_ratio=0.):
        super(Embedding, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2  # Patch数量

        self.patch_conv = nn.Conv2d(image_channels, dim, patch_size, patch_size)  # 使用卷积将图像划分成Patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))  # class embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))  # position embedding
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x):
        x = self.patch_conv(x)
        x = rearrange(x, "B C H W -> B (H W) C")
        cls_token = torch.repeat_interleave(self.cls_token, x.shape[0], dim=0)  # (1,1,dim) -> (B,1,dim)
        x = torch.cat([cls_token, x], dim=1)  # (B,1,dim) cat (B,num_patches,dim) --> (B,num_patches+1,dim)
        a = self.pos_emb
        x = x + self.pos_emb

        return self.dropout(x)  # token


class MultiHeadAttention(nn.Module):  # Multi-Head Attention
    def __init__(self, dim, num_heads=6, drop_ratio=0.):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)  # 使用一个Linear，计算得到qkv
        self.dropout = nn.Dropout(drop_ratio)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # B: Batch Size / P: Num of Patches / D: Dim of Patch / H: Num of Heads / d: Dim of Head
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "B P (C H d) -> C B H P d", C=3, H=self.num_heads, d=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离qkv
        k = rearrange(k, "B H P d -> B H d P")
        # Attention(Q, K, V ) = softmax(QKT/dk)V （T表示转置)
        attn = torch.matmul(q, k) * self.head_dim ** -0.5  # QKT/dk
        attn = F.softmax(attn, dim=-1)  # softmax(QKT/dk)
        attn = self.dropout(attn)
        x = torch.matmul(attn, v)  # softmax(QKT/dk)V
        x = rearrange(x, "B H P d -> B P (H d)")
        x = self.proj(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):  # MLP
    def __init__(self, in_dims, hidden_dims=None, drop_ratio=0.):
        super(MLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = in_dims * 4  # linear的hidden_dims默认为in_dims的4倍

        self.fc1 = nn.Linear(in_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, in_dims)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x):
        # Linear + GELU + Dropout + Linear + Dropout
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):  # Transformer Encoder Block
    def __init__(self, dim, num_heads=6, drop_ratio=0.):
        super(EncoderBlock, self).__init__()

        self.layernorm1 = nn.LayerNorm(dim)
        self.multiheadattn = MultiHeadAttention(dim, num_heads)
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        # 两次残差连接，分别在Multi-Head Attention和MLP之后
        x0 = x
        x = self.layernorm1(x)
        x = self.multiheadattn(x)
        x = self.dropout(x)
        x1 = x + x0  # 第一次残差连接
        x = self.layernorm2(x1)
        x = self.mlp(x)
        x = self.dropout(x)
        return x + x1  # 第二次残差连接


class MLPHead(nn.Module):  # MLP Head
    def __init__(self, dim, num_classes=1000):
        super(MLPHead, self).__init__()
        self.layernorm = nn.LayerNorm(dim)
        # 对于一般数据集，此处为1层Linear; 对于ImageNet-21k数据集，此处为Linear+Tanh+Linear
        self.mlphead = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.layernorm(x)
        cls = x[:, 0, :]  # 去除class token
        return self.mlphead(cls)


class ViT(nn.Module):  # Vision Transformer
    def __init__(self, image_channels=3, image_size=224, num_classes=1000, patch_size=16, dim=512, num_heads=6,
                 layers=1):
        super(ViT, self).__init__()
        self.embedding = Embedding(image_channels, image_size, patch_size, dim)
        self.encoder = nn.Sequential(
            *[EncoderBlock(dim, num_heads) for i in range(layers)]  # encoder结构为layers(L)个Transformer Encoder Block
        )
        self.head = MLPHead(dim, num_classes)

    def forward(self, x):
        x_emb = self.embedding(x)
        feature = self.encoder(x_emb)
        return self.head(feature)


def vit_base(num_classes=256):  # ViT-Base
    return ViT(image_channels=16, image_size=224, num_classes=num_classes, patch_size=8, dim=768, num_heads=6,
               layers=1)