import math
import torch
import torch.nn as nn
import torchvision.models as models


def modify_resnet_conv1(model: nn.Module, in_channels: int) -> nn.Module:
    old_conv = model.conv1
    old_w = old_conv.weight.data.clone()  # (64, 3, 7, 7)

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    with torch.no_grad():
        if in_channels == 3:
            new_conv.weight.copy_(old_w)
        elif in_channels == 2:
            mean_w = old_w.mean(dim=1, keepdim=True)
            new_conv.weight[:, 0:1, :, :].copy_(mean_w)
            new_conv.weight[:, 1:2, :, :].copy_(mean_w)
        else:
            raise ValueError(f"Unsupported in_channels={in_channels}. Expected 2 or 3.")

    model.conv1 = new_conv
    return model


class DSDL(nn.Module):
    """Deep Semantic Dictionary Learning with dual-branch visual encoder.

    Visual encoder:
        Optical (4ch) -> ResNet101 -> 2048-d feature
        SAR (1ch)     -> ResNet101 -> 2048-d feature
        Fuse: f = f_opt + f_sar  (按导师要求：ResNet 输出后加和)

    The semantic dictionary part (W1/W2 + closed-form alpha) keeps the same.
    """

    def __init__(self, base_model_opt, base_model_sar, num_classes, alpha, in_channel=300):
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes

        # Optical backbone
        self.features_opt = nn.Sequential(
            base_model_opt.conv1,
            base_model_opt.bn1,
            base_model_opt.relu,
            base_model_opt.maxpool,
            base_model_opt.layer1,
            base_model_opt.layer2,
            base_model_opt.layer3,
            base_model_opt.layer4,
        )

        # SAR backbone
        self.features_sar = nn.Sequential(
            base_model_sar.conv1,
            base_model_sar.bn1,
            base_model_sar.relu,
            base_model_sar.maxpool,
            base_model_sar.layer1,
            base_model_sar.layer2,
            base_model_sar.layer3,
            base_model_sar.layer4,
        )

        self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        # ===== semantic dictionary params (same as before) =====
        self.W1 = nn.Parameter(torch.zeros(size=(in_channel, 1024)))
        stdv1 = 1.0 / math.sqrt(self.W1.size(1))
        self.W1.data.uniform_(-stdv1, stdv1)

        self.relu = nn.LeakyReLU(0.2)

        self.W2 = nn.Parameter(torch.zeros(size=(1024, 2048)))
        stdv2 = 1.0 / math.sqrt(self.W2.size(1))
        self.W2.data.uniform_(-stdv2, stdv2)

        # Keep interface
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, optical, sar, semantic_vectors):
        """Forward.

        optical: Tensor [B, 4, H, W]
        sar:     Tensor [B, 1, H, W]
        semantic_vectors: Tensor [C, D] or [B, C, D]
        """
        # semantic_vectors: [C, D]
        if semantic_vectors.dim() == 3:
            semantic_vectors = semantic_vectors[0]
        if semantic_vectors.dim() != 2:
            raise ValueError(
                f"语义向量应为2维 [num_classes, embedding_dim]，但得到了 {semantic_vectors.shape}"
            )

        if optical.size(1) != 3:
            raise ValueError(f"Optical 分支期望 3 通道输入，但得到了 {optical.size(1)}")
        if sar.size(1) != 2:
            raise ValueError(f"SAR 分支期望 2 通道输入，但得到了 {sar.size(1)}")

        # ===== dual visual encoder =====
        f_opt = self.features_opt(optical)
        f_opt = self.pooling(f_opt).view(f_opt.size(0), -1)  # [B, 2048]

        f_sar = self.features_sar(sar)
        f_sar = self.pooling(f_sar).view(f_sar.size(0), -1)  # [B, 2048]

        # 直接加和
        feature = f_opt + f_sar

        # ===== semantic dictionary (same) =====
        semantic = torch.matmul(semantic_vectors, self.W1)  # [C,1024]
        semantic = self.relu(semantic)
        semantic = torch.matmul(semantic, self.W2)          # [C,2048]

        res_semantic = torch.matmul(semantic, self.W2.transpose(0, 1))
        res_semantic = self.relu(res_semantic)
        res_semantic = torch.matmul(res_semantic, self.W1.transpose(0, 1))

        device = semantic.device
        eye_matrix = self.alpha * torch.eye(self.num_classes, device=device)

        score = torch.matmul(
            torch.inverse(torch.matmul(semantic, semantic.transpose(0, 1)) + eye_matrix),
            torch.matmul(semantic, feature.transpose(0, 1))
        ).transpose(0, 1)

        return score, semantic_vectors, res_semantic, feature, semantic

    def get_config_optim(self, lr, lrp):
        # Important: optimize both backbones
        return [
            {'params': self.features_opt.parameters(), 'lr': lr * lrp},
            {'params': self.features_sar.parameters(), 'lr': lr * lrp},
            {'params': self.W1, 'lr': lr},
            {'params': self.W2, 'lr': lr},
        ]


def load_model(num_classes, alpha, pretrained=True, in_channel=300):
    """Build dual-branch model.

    - Optical branch: ResNet101 with conv1=4ch
    - SAR branch:     ResNet101 with conv1=1ch
    """
    if pretrained:
        weights = models.ResNet101_Weights.IMAGENET1K_V1
        base_opt = models.resnet101(weights=weights)
        base_sar = models.resnet101(weights=weights)
    else:
        base_opt = models.resnet101(weights=None)
        base_sar = models.resnet101(weights=None)

    base_opt = modify_resnet_conv1(base_opt, in_channels=3)
    base_sar = modify_resnet_conv1(base_sar, in_channels=2)

    return DSDL(
        base_model_opt=base_opt,
        base_model_sar=base_sar,
        num_classes=num_classes,
        alpha=alpha,
        in_channel=in_channel,
    )


__all__ = ['DSDL', 'load_model']
