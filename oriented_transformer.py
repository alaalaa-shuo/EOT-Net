import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from ssim_compute import compute_distance_matrix

import einops
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def OSP(x, vector):

    # B, N, D = x.shape
    x = x.squeeze(0)
    x = x.T
    # D*N
    V_selected = vector / torch.norm(vector)
    P_sub = V_selected.T @ V_selected
    # D*D
    X_projected = P_sub @ x
    # D*N
    x = X_projected.T.unsqueeze(0)
    # B*D*N

    return x


# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"Dim should be divisible by heads dim={dim}, heads={num_heads}"
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, self.num_heads*dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, self.num_heads*dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, self.num_heads*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.transform_layer = nn.Sequential(
            nn.Linear(self.num_heads, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, N, C = x.shape
        # B1C -> B1HC -> BH1C
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C).permute(0, 2, 1, 3)
        # BNC -> BNHC -> BHNC
        k = self.wk(x).reshape(B, N, self.num_heads, C).permute(0, 2, 1, 3)
        # BNC -> BNHC -> BHNC
        v = self.wv(x).reshape(B, N, self.num_heads, C).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1C @ BHCN -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_heads = attn @ v
        # (BH1N @ BHNC) -> BH1C

        # redundancy assessing
        sim_matrix = compute_distance_matrix(x_heads)
        # H*H
        mask_redundancy = self.transform_layer(sim_matrix)
        # H*1
        mask_redundancy = 1 - F.softmax(mask_redundancy, dim=0)
        mask_redundancy = mask_redundancy.unsqueeze(0).unsqueeze(2)
        # 1*H*1*1

        x_valuable = x_heads * mask_redundancy
        x = torch.sum(x_valuable, dim=1).squeeze(1)
        # B*H*1*C -> B*1*1*C -> B*1*C

        # x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=3., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.15, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        edm_token, attn = self.attn(x)
        x = x[:, 0:1, ...] + self.drop_path(edm_token)  # Better result
        # B*1*C
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CrossAttentionBlock(dim, num_heads=heads, drop=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        attn_weights = []
        for attn, ff in self.layers:
            edm_token, attn_weight = attn(x)
            attn_weights.append(attn_weight)
            x = torch.cat((edm_token, self.norm(x[:, 1:, :])), dim=1)
            x = ff(x) + x

        return x, attn_weights[-1]


class OrientedTrans(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth=2, heads, mlp_dim,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size. '

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            # nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Sequential(
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                      h=image_height // patch_height, w=image_width // patch_width, p1=patch_height, p2=patch_width)
        )

    def forward(self, img, edm_vca_emb, idx):
        # B*D*H*W
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_token = edm_vca_emb[:, idx].unsqueeze(1).T
        cls_token = cls_token.unsqueeze(0)
        # 1*1*D
        cls_tokens = repeat(cls_token, '() n d -> b n d', b=b)  # batch_size个cls_tokens

        x = torch.cat((cls_tokens, x), dim=1)
        x = OSP(x, cls_token.squeeze(0))

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        x, attn_weight = self.transformer(x)
        # B*(N+1)*D, B*Head*1*(N+1)
        attn_weight = attn_weight.transpose(2, 3).squeeze(3)
        attn_weight = attn_weight[:, :, 1:].transpose(1, 2)
        # B*N*Head

        x = x[:,1:,:]
        # B*N*D
        x = self.to_latent(x)
        # B*D*H*W
        return x, attn_weight

