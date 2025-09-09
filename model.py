import torch
import torch.nn as nn

from oriented_transformer import OrientedTrans
from spatial_complementary import SCM
from ChannelShuffle import channel_shuffle
from utils import Swish, LayerNormProxy


class AutoEncoder(nn.Module):
    def __init__(self, P, L, size, patch, dim, init_bundle, num_bundle=10):
        super(AutoEncoder, self).__init__()
        self.P, self.L, self.size, self.dim, self.patch = P, L, size, dim, patch
        self.num_bundle = num_bundle
        self.bundles = nn.Parameter(init_bundle, requires_grad=False)

        self.patch_proj = nn.Sequential(
            nn.Conv2d(self.L, self.dim, 1, 1, 0),
            LayerNormProxy(self.dim)
        )

        self.intra_weight_layers = nn.ModuleList()
        for _ in range(P):
            self.intra_weight_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.dim, out_channels=self.num_bundle, kernel_size=1, stride=1, padding=0),
                    nn.Softmax(dim=1)
                    )
            )

        self.bundle_est_layers = nn.ModuleList()
        for _ in range(P):
            self.bundle_est_layers.append(
                nn.Sequential(
                    nn.Linear(4, 1),
                    Swish()
                )
            )

        self.encoder_abd = nn.Sequential(
            nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.Conv2d(in_channels=self.dim, out_channels=self.P, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )

        self.otrans_layers = nn.ModuleList()
        for _ in range(P):
            self.otrans_layers.append(OrientedTrans(image_size=self.size, patch_size=self.patch, dim=self.dim, heads=4,
                                    mlp_dim=2*self.dim, dropout=0.0, emb_dropout=0.0)
                                     )

        self.scms = nn.ModuleList()
        for i in range(self.P):
            self.scms.append(SCM(num_levels=self.P, in_channels=self.dim, level=i))

        self.channel_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.P*self.dim, out_channels=self.dim, kernel_size=1, groups=4),
            nn.GELU()
        )

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, edm_vca):

        x_emb = self.patch_proj(x)
        # B*D*H*W
        edm_vca_emb = self.patch_proj(edm_vca.unsqueeze(0).unsqueeze(3))
        # 1*D*P*1
        edm_vca_emb = edm_vca_emb.squeeze(3).squeeze(0)

        edm_features = []
        bundle_attn_scores = []
        for i, otrans in enumerate(self.otrans_layers):
            edm_feature, atten_weight = otrans(x_emb, edm_vca_emb, i)
            edm_features.append(edm_feature)
            bundle_attn_scores.append(atten_weight)

        enhanced_features = []
        for i, scm in enumerate(self.scms):
            enhanced_feature = scm(edm_features)
            enhanced_features.append(enhanced_feature)

        intra_weights = []
        for i, encoder_intra in enumerate(self.intra_weight_layers):
            intra_weight = encoder_intra(edm_features[i])
            # B*num_bundle*H*W
            intra_weight = intra_weight.flatten(start_dim=-2).squeeze(0)
            # B*num_bundle*N -> num_bundle*N
            intra_weights.append(intra_weight)

        enhanced_features = torch.cat(enhanced_features, dim=1)
        # B*(P*dim)*H*W: concatenate follow the channel dim
        enhanced_features = channel_shuffle(enhanced_features, groups=4)
        fused_feature = self.channel_fuse(enhanced_features)
        # B*dim*H*W

        weighted_bundles = []
        for i in range(self.P):
            # bundle = self.bundles[i, :, :]
            # L*num_bundle
            intra_weight = intra_weights[i]
            # num_bundle*N
            weighted_bundle = self.bundles[i, :, :] @ intra_weight
            # L*N
            weighted_bundle = weighted_bundle.unsqueeze(0)
            # 1*L*N
            weighted_bundles.append(weighted_bundle)

        weighted_endmember = torch.cat(weighted_bundles, dim=0)
        # P*L*N

        abu_est = self.encoder_abd(fused_feature)
        # B*P*H*W

        abu_est = abu_est.flatten(start_dim=-2).squeeze(0).transpose(0, 1).unsqueeze(2)
        # N*P*1

        endmember = weighted_endmember.permute(2, 1, 0)
        # N*L*P
        re_result = (endmember @ abu_est).squeeze(2)
        # N*L*1 -> N*L
        abu_est = abu_est.squeeze(2)
        # N*P
        re_result = re_result.unsqueeze(0)
        # B*N*L
        # endmember = endmember.squeeze(0)
        # N*L*P
        return abu_est, re_result, endmember

