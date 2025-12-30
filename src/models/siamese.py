from torch import nn
import torch.nn.functional as F
import torch
class FPNWrapper(nn.Module):
    def __init__(self, backbone, out_channels=256):
        super().__init__()
        # Assume backbone returns feature maps from 3 stages
        self.backbone = backbone
        self.lateral3 = nn.Conv2d(48, out_channels, 1)
        self.lateral4 = nn.Conv2d(80, out_channels, 1)
        self.lateral5 = nn.Conv2d(960, out_channels, 1)
        self.smoothing3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smoothing4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smoothing5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        # Example: features from conv3, conv4, conv5
        c3, c4, c5 = self.backbone(x)
        p5 = self.lateral5(c5)
        p5 = self.smoothing5(p5)
        p4 = self.lateral4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p4 = self.smoothing4(p4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p3 = self.smoothing3(p3)
        return [p3, p4, p5]

class AttentionPooling(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, feat):
        attn_map = self.attn(feat)
        weighted = (feat * attn_map).sum(dim=(2, 3)) / (attn_map.sum(dim=(2, 3)) + 1e-6)
        return weighted, attn_map

class PrototypeExtractor(nn.Module):
    def __init__(self, fpn, attn_pool, output_size=(7, 7)):
        super().__init__()
        self.fpn = fpn
        self.attn_pool = attn_pool
        self.output_size = output_size

    def forward(self, support_patches: torch.Tensor):
        """
        Args:
            support_patches: Tensor of shape [B, C, H, W]
                             where B is the number of support shots.
        Returns:
            proto: Tensor of shape [B, C_out] (The averaged prototype vector)
        """

        # Safety check for empty input
        if support_patches.size(0) == 0:
            return None


        fpn_feats = self.fpn(support_patches)

        roi_feat = fpn_feats[-1]
        if roi_feat.shape[-2:] != self.output_size:
             roi_feat = F.adaptive_avg_pool2d(roi_feat, self.output_size)

        proto, _ = self.attn_pool(roi_feat)


        return proto

class Backbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


    def forward(self, x):
        # Return feature maps from 3 stages
        c3, c4, c5 = self.model(x)
        return c3, c4, c5

    def destroy_hooks(self):
        # Call this method to remove hooks when done
        self.handle1.remove()
        self.handle2.remove()
        self.handle3.remove()