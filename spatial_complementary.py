import torch
import torch.nn as nn


class SCM(nn.Module):
    def __init__(self,
                 num_levels,
                 in_channels,
                 level,
                 ):
        super(SCM, self).__init__()

        self.num_levels = num_levels
        self.in_channels = in_channels
        self.level = level

        self.sc_convs = nn.ModuleList()
        for i in range(num_levels):
            self.sc_convs.append(
                nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1),
                nn.Sigmoid())
            )

    def forward(self, features):

        assert len(features) == self.num_levels, "The number of input feature maps does not match num_levels."

        masked_features = []
        selective_feature = torch.zeros_like(features[0])
        spatial_selective_map = torch.zeros_like(features[0])

        for i in range(self.num_levels):
            if i == self.level:
                spatial_selective_map = self.sc_convs[i](features[i])
                selective_feature = spatial_selective_map * features[i]
            else:
                mask = self.sc_convs[i](features[i])
                masked_feature = mask * features[i]
                masked_features.append(masked_feature)

        masked_features = torch.stack(masked_features)  # shape will be 3*B*D*H*W
        complementary_feature = torch.sum(masked_features, dim=0)
        enhanced_feature = features[self.level] + selective_feature + (1-spatial_selective_map) * complementary_feature

        return enhanced_feature


