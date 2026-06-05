import inspect
import io
import logging
import os
import sys
from urllib.parse import urlparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from torchvision.models import mobilenet_v3_large, resnet50, resnet101
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    fcn_resnet50,
    fcn_resnet101,
)


class _NullConsole(io.TextIOBase):
    """Fallback stream for GUI/pythonw environments where stdio can be None."""

    def writable(self):
        return True

    def write(self, text):
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False


def _ensure_console_streams():
    if sys.stdout is None:
        sys.stdout = _NullConsole()
    if sys.stderr is None:
        sys.stderr = _NullConsole()


def _offline_mode_enabled():
    return bool(config.get("offline_mode", True))


def _configure_offline_environment():
    if not _offline_mode_enabled():
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _cached_pretrained_enabled():
    return bool(config.get("use_cached_pretrained", True))


def _torchvision_weight_cache_path(weights):
    url = getattr(weights, "url", None)
    if not url:
        return None
    filename = os.path.basename(urlparse(url).path)
    if not filename:
        return None
    return os.path.join(torch.hub.get_dir(), "checkpoints", filename)


def _allow_torchvision_weights(weights, label):
    _configure_offline_environment()
    if weights is None:
        return None
    if not _offline_mode_enabled():
        return weights
    if not _cached_pretrained_enabled():
        logging.info(f"{label} 离线模式已启用，跳过预训练权重")
        return None

    cache_path = _torchvision_weight_cache_path(weights)
    if cache_path and os.path.exists(cache_path):
        logging.info(f"{label} 使用本地缓存预训练权重: {cache_path}")
        return weights

    logging.warning(f"{label} 离线模式下未找到本地预训练缓存，改用随机初始化")
    return None


def _build_with_pretrained_fallback(builder, label, random_kwargs=None, **kwargs):
    random_kwargs = random_kwargs or {}
    try:
        return builder(**kwargs)
    except Exception as e:
        if kwargs.get("weights") is None:
            raise
        logging.warning(f"{label} 预训练权重加载失败，改用随机初始化: {e}")
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.update(random_kwargs)
        fallback_kwargs["weights"] = None
        if "weights_backbone" in fallback_kwargs:
            fallback_kwargs["weights_backbone"] = None
        return builder(**fallback_kwargs)


ARCH_TO_BACKBONES = {
    "DeepLabV3": ["ResNet50", "ResNet101", "MobileNetV3-Large"],
    "DeepLabV3+": ["ResNet50", "ResNet101", "MobileNetV3-Large"],
    "FCN": ["ResNet50", "ResNet101"],
    "PSPNet": ["ResNet50"],
    "UPerNet-Swin": ["Tiny"],
    "U-Net": [],
    "SFA-Net": [],
}

ARCH_PARAM_SCHEMAS = {
    "DeepLabV3": [
        {"key": "aux_loss", "label": "辅助损失", "type": "bool", "default": False},
    ],
    "DeepLabV3+": [
        {"key": "output_stride", "label": "输出步长", "type": "int", "default": 16, "options": [8, 16]},
        {"key": "aspp_rate1", "label": "ASPP率-1", "type": "int", "default": 6},
        {"key": "aspp_rate2", "label": "ASPP率-2", "type": "int", "default": 12},
        {"key": "aspp_rate3", "label": "ASPP率-3", "type": "int", "default": 18},
        {"key": "decoder_channels", "label": "解码通道", "type": "int", "default": 256},
        {"key": "low_level_channels", "label": "低层投影通道", "type": "int", "default": 48},
    ],
    "FCN": [
        {"key": "aux_loss", "label": "辅助损失", "type": "bool", "default": False},
    ],
    "PSPNet": [],
    "UPerNet-Swin": [],
    "U-Net": [
        {"key": "base_channels", "label": "基础通道数", "type": "int", "default": 64},
    ],
    "SFA-Net": [
        {"key": "base_channels", "label": "基础通道数", "type": "int", "default": 64},
    ],
}

MODEL_BUILDERS = {
    "DeepLabV3 (ResNet50)": (deeplabv3_resnet50, "deeplabv3", 256),
    "DeepLabV3 (ResNet101)": (deeplabv3_resnet101, "deeplabv3", 256),
    "DeepLabV3 (MobileNetV3-Large)": (deeplabv3_mobilenet_v3_large, "deeplabv3", 256),
    "DeepLabV3+ (ResNet50)": (None, "deeplabv3plus", 256),
    "DeepLabV3+ (ResNet101)": (None, "deeplabv3plus", 256),
    "DeepLabV3+ (MobileNetV3-Large)": (None, "deeplabv3plus", 256),
    "FCN (ResNet50)": (fcn_resnet50, "fcn", 512),
    "FCN (ResNet101)": (fcn_resnet101, "fcn", 512),
    "PSPNet (ResNet50)": (None, "pspnet", 2048),
    "UPerNet-Swin (Tiny)": (None, "upernet_swin", 512),
    "U-Net": (None, "unet", 64),
    "SFA-Net": (None, "sfanet", 64),
}


def get_architectures():
    return list(ARCH_TO_BACKBONES.keys())


def architecture_requires_backbone(architecture):
    return len(ARCH_TO_BACKBONES.get(architecture, [])) > 0


def get_backbones(architecture):
    return ARCH_TO_BACKBONES.get(architecture, [])


def get_default_backbone(architecture):
    candidates = get_backbones(architecture)
    return candidates[0] if candidates else None


def get_architecture_param_schema(architecture):
    return ARCH_PARAM_SCHEMAS.get(architecture, [])


def compose_model_name(architecture, backbone=None):
    if architecture_requires_backbone(architecture):
        if not backbone:
            backbone = get_default_backbone(architecture)
        return f"{architecture} ({backbone})"
    return architecture


def split_model_name(model_name):
    if " (" in model_name and model_name.endswith(")"):
        arch, backbone = model_name[:-1].split(" (", 1)
        return arch, backbone
    return model_name, None


def get_default_model_name():
    return compose_model_name("DeepLabV3", "ResNet50")


def normalize_architecture_params(architecture, raw_params):
    normalized = {}
    schema = get_architecture_param_schema(architecture)
    raw_params = raw_params or {}

    for item in schema:
        key = item["key"]
        vtype = item["type"]
        default = item["default"]
        value = raw_params.get(key, default)
        try:
            if vtype == "bool":
                if isinstance(value, str):
                    normalized[key] = value.strip().lower() in ("1", "true", "yes", "on")
                else:
                    normalized[key] = bool(value)
            elif vtype == "int":
                normalized[key] = int(value)
            elif vtype == "float":
                normalized[key] = float(value)
            else:
                normalized[key] = value
        except Exception:
            normalized[key] = default

    # Clamp to safe ranges
    if architecture in ("U-Net", "SFA-Net"):
        normalized["base_channels"] = max(16, min(512, normalized.get("base_channels", 64)))
    if architecture == "DeepLabV3+":
        normalized["output_stride"] = 8 if normalized.get("output_stride", 16) == 8 else 16
        normalized["aspp_rate1"] = max(1, normalized.get("aspp_rate1", 12))
        normalized["aspp_rate2"] = max(1, normalized.get("aspp_rate2", 24))
        normalized["aspp_rate3"] = max(1, normalized.get("aspp_rate3", 36))
        normalized["decoder_channels"] = max(64, min(512, normalized.get("decoder_channels", 256)))
        normalized["low_level_channels"] = max(16, min(256, normalized.get("low_level_channels", 48)))
    return normalized


def _resolve_seg_weights(model_name, use_pretrained):
    if not use_pretrained:
        return None
    try:
        if model_name == "DeepLabV3 (ResNet50)":
            from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights
            return _allow_torchvision_weights(DeepLabV3_ResNet50_Weights.DEFAULT, model_name)
        if model_name == "DeepLabV3 (ResNet101)":
            from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet101_Weights
            return _allow_torchvision_weights(DeepLabV3_ResNet101_Weights.DEFAULT, model_name)
        if model_name == "DeepLabV3 (MobileNetV3-Large)":
            from torchvision.models.segmentation.deeplabv3 import DeepLabV3_MobileNet_V3_Large_Weights
            return _allow_torchvision_weights(DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT, model_name)
        if model_name == "FCN (ResNet50)":
            from torchvision.models.segmentation.fcn import FCN_ResNet50_Weights
            return _allow_torchvision_weights(FCN_ResNet50_Weights.DEFAULT, model_name)
        if model_name == "FCN (ResNet101)":
            from torchvision.models.segmentation.fcn import FCN_ResNet101_Weights
            return _allow_torchvision_weights(FCN_ResNet101_Weights.DEFAULT, model_name)
    except ImportError:
        return None
    return None


def _resolve_backbone_weights(backbone_name, use_pretrained):
    if not use_pretrained:
        return None
    try:
        if backbone_name == "ResNet50":
            from torchvision.models import ResNet50_Weights
            return _allow_torchvision_weights(ResNet50_Weights.DEFAULT, backbone_name)
        if backbone_name == "ResNet101":
            from torchvision.models import ResNet101_Weights
            return _allow_torchvision_weights(ResNet101_Weights.DEFAULT, backbone_name)
        if backbone_name == "MobileNetV3-Large":
            from torchvision.models import MobileNet_V3_Large_Weights
            return _allow_torchvision_weights(MobileNet_V3_Large_Weights.DEFAULT, backbone_name)
    except ImportError:
        return None
    return None


def replace_classifier_head(base_model, family, cls_in_channels, num_classes):
    if family == "deeplabv3":
        if hasattr(base_model.classifier, "__getitem__"):
            base_model.classifier[-1] = nn.Conv2d(cls_in_channels, num_classes, 1, 1)
        else:
            base_model.classifier = nn.Conv2d(cls_in_channels, num_classes, 1, 1)
    elif family == "fcn":
        if hasattr(base_model.classifier, "__getitem__"):
            base_model.classifier[4] = nn.Conv2d(cls_in_channels, num_classes, 1, 1)
        else:
            base_model.classifier = nn.Conv2d(cls_in_channels, num_classes, 1, 1)
    else:
        raise ValueError(f"不支持的模型族: {family}")


def _get_out_channels(module):
    if hasattr(module, "out_channels"):
        return module.out_channels
    for sub_module in reversed(list(module.modules())):
        if hasattr(sub_module, "out_channels"):
            return sub_module.out_channels
    raise ValueError("无法推断模块输出通道数")


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for module in self:
            x = module(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super().__init__()
        branches = [
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        ]
        branches.extend(ASPPConv(in_channels, out_channels, r) for r in atrous_rates)
        branches.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(len(branches) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )


    def forward(self, x):
        return self.project(torch.cat([conv(x) for conv in self.convs], dim=1))


class DeepLabV3PlusHead(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_rates=(12, 24, 36), decoder_channels=256,
                 low_level_proj_channels=48):
        super().__init__()
        self.aspp = ASPP(in_channels, decoder_channels, atrous_rates=aspp_rates)
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_level_channels, low_level_proj_channels, 1, bias=False),
            nn.BatchNorm2d(low_level_proj_channels),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_channels + low_level_proj_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(decoder_channels, num_classes, 1)

    def forward(self, features):
        low = self.low_level_proj(features["low_level"])
        high = self.aspp(features["out"])
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.classifier(self.decoder(torch.cat([low, high], dim=1)))


class ResNetBackbone(nn.Module):
    def __init__(self, backbone_name, use_pretrained, output_stride=16):
        super().__init__()
        weights = _resolve_backbone_weights(backbone_name, use_pretrained)
        dilation_cfg = [False, True, True] if output_stride == 8 else [False, False, True]
        if backbone_name == "ResNet50":
            model = _build_with_pretrained_fallback(
                resnet50,
                backbone_name,
                weights=weights,
                progress=False,
                replace_stride_with_dilation=dilation_cfg,
            )
        elif backbone_name == "ResNet101":
            model = _build_with_pretrained_fallback(
                resnet101,
                backbone_name,
                weights=weights,
                progress=False,
                replace_stride_with_dilation=dilation_cfg,
            )
        else:
            raise ValueError(f"不支持的ResNet骨干: {backbone_name}")
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = model.layer1, model.layer2, model.layer3, model.layer4
        self.out_channels = 2048
        self.low_level_channels = 256

    def forward(self, x):
        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        return {"low_level": low, "out": self.layer4(x)}


class MobileNetV3Backbone(nn.Module):
    def __init__(self, use_pretrained):
        super().__init__()
        model = _build_with_pretrained_fallback(
            mobilenet_v3_large,
            "MobileNetV3-Large",
            weights=_resolve_backbone_weights("MobileNetV3-Large", use_pretrained),
            progress=False,
        )
        features = model.features
        stage_indices = [0] + [i for i, b in enumerate(features) if getattr(b, "_is_cn", False)] + [len(features) - 1]
        low_idx = stage_indices[-4]
        self.low_features = features[: low_idx + 1]
        self.high_features = features[low_idx + 1 :]
        self.low_level_channels = _get_out_channels(features[low_idx])
        self.out_channels = _get_out_channels(features[-1])

    def forward(self, x):
        low = self.low_features(x)
        return {"low_level": low, "out": self.high_features(low)}


class DeepLabV3PlusModel(nn.Module):
    def __init__(self, backbone_name, num_classes, use_pretrained, config_params=None):
        super().__init__()
        config_params = config_params or {}
        output_stride = config_params.get("output_stride", 16)
        aspp_rates = (
            config_params.get("aspp_rate1", 12),
            config_params.get("aspp_rate2", 24),
            config_params.get("aspp_rate3", 36),
        )
        decoder_channels = config_params.get("decoder_channels", 256)
        low_proj_ch = config_params.get("low_level_channels", 48)

        if backbone_name in ("ResNet50", "ResNet101"):
            self.backbone = ResNetBackbone(backbone_name, use_pretrained, output_stride=output_stride)
        else:
            self.backbone = MobileNetV3Backbone(use_pretrained)
        self.classifier = DeepLabV3PlusHead(
            self.backbone.out_channels,
            self.backbone.low_level_channels,
            num_classes,
            aspp_rates=aspp_rates,
            decoder_channels=decoder_channels,
            low_level_proj_channels=low_proj_ch,
        )

    def forward(self, x):
        out = self.classifier(self.backbone(x))
        return {"out": F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)}


class TensorOutputWrapper(nn.Module):
    """Wrap third-party segmentation modules so training can always consume {'out': logits}."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            logits = out["out"] if "out" in out else out.get("logits")
        elif hasattr(out, "logits"):
            logits = out.logits
        else:
            logits = out
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"out": logits}


def _missing_dependency_error(model_name, package_name, install_command):
    raise ImportError(
        f"{model_name} 需要安装 {package_name}。请先执行: {install_command}"
    )


def build_pspnet_resnet50(num_classes, use_pretrained):
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        _missing_dependency_error("PSPNet-ResNet50", "segmentation_models_pytorch", "pip install segmentation-models-pytorch")
    encoder_weights = "imagenet" if use_pretrained and not _offline_mode_enabled() else None
    if use_pretrained and encoder_weights is None:
        logging.warning("PSPNet 离线模式下跳过 encoder 预训练权重，改用随机初始化")
    try:
        model = smp.PSPNet(
            encoder_name="resnet50",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
    except Exception as e:
        if encoder_weights is None:
            raise
        logging.warning(f"PSPNet 预训练权重加载失败，改用随机初始化: {e}")
        model = smp.PSPNet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
        )
    return TensorOutputWrapper(model)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class MSCABlock(nn.Module):
    """SegNeXt-style multi-scale convolutional attention block."""

    def __init__(self, channels, mlp_ratio=4, drop=0.0):
        super().__init__()
        hidden_channels = channels * mlp_ratio
        self.norm = nn.BatchNorm2d(channels)
        self.proj1 = nn.Conv2d(channels, hidden_channels, 1)
        self.dw_conv = nn.Conv2d(hidden_channels, hidden_channels, 5, padding=2, groups=hidden_channels)
        self.branch7 = nn.Conv2d(hidden_channels, hidden_channels, (1, 7), padding=(0, 3), groups=hidden_channels)
        self.branch11 = nn.Conv2d(hidden_channels, hidden_channels, (7, 1), padding=(3, 0), groups=hidden_channels)
        self.branch21 = nn.Conv2d(hidden_channels, hidden_channels, (1, 11), padding=(0, 5), groups=hidden_channels)
        self.branch31 = nn.Conv2d(hidden_channels, hidden_channels, (11, 1), padding=(5, 0), groups=hidden_channels)
        self.attn = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.proj2 = nn.Conv2d(hidden_channels, channels, 1)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.proj1(self.norm(x))
        x = F.gelu(x)
        base = self.dw_conv(x)
        attn = base + self.branch11(self.branch7(base)) + self.branch31(self.branch21(base))
        x = self.proj2(self.attn(attn) * x)
        return residual + self.drop(x)


def build_upernet_swin_t(num_classes, use_pretrained):
    try:
        from transformers import SwinConfig, UperNetConfig, UperNetForSemanticSegmentation
    except ImportError:
        _missing_dependency_error("UPerNet-Swin-Tiny", "transformers", "pip install transformers")

    if use_pretrained:
        try:
            model = UperNetForSemanticSegmentation.from_pretrained(
                "openmmlab/upernet-swin-tiny",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                local_files_only=_offline_mode_enabled(),
            )
            return TensorOutputWrapper(model)
        except Exception as e:
            logging.warning(f"加载UPerNet-Swin预训练权重失败，改用随机初始化: {e}")

    backbone_config = SwinConfig(
        image_size=224,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )
    config = UperNetConfig(backbone_config=backbone_config, num_labels=num_classes)
    return TensorOutputWrapper(UperNetForSemanticSegmentation(config))


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base_channels=64):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.enc1 = self._conv_block(in_channels, c1)
        self.enc2 = self._conv_block(c1, c2)
        self.enc3 = self._conv_block(c2, c3)
        self.enc4 = self._conv_block(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._conv_block(c4, c4 * 2)
        self.up4, self.dec4 = nn.ConvTranspose2d(c4 * 2, c4, 2, 2), self._conv_block(c4 * 2, c4)
        self.up3, self.dec3 = nn.ConvTranspose2d(c4, c3, 2, 2), self._conv_block(c3 * 2, c3)
        self.up2, self.dec2 = nn.ConvTranspose2d(c3, c2, 2, 2), self._conv_block(c2 * 2, c2)
        self.up1, self.dec1 = nn.ConvTranspose2d(c2, c1, 2, 2), self._conv_block(c1 * 2, c1)
        self.classifier = nn.Conv2d(c1, num_classes, 1)

    @staticmethod
    def _conv_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    @staticmethod
    def _align(src, ref):
        return F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False) if src.shape[-2:] != ref.shape[-2:] else src

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3)); b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self._align(self.up4(b), e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._align(self.up3(d4), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._align(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._align(self.up1(d2), e1), e1], dim=1))
        return {"out": self.classifier(d1)}


class SFABlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels // 4, 1), nn.ReLU(inplace=True), nn.Conv2d(channels // 4, channels, 1), nn.Sigmoid())
        self.spatial_gate = nn.Sequential(nn.Conv2d(channels, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x):
        x = x * self.channel_gate(x)
        return x * self.spatial_gate(x)


class SFANet(nn.Module):
    def __init__(self, num_classes=2, base_channels=64):
        super().__init__()
        self.encoder1 = SimpleUNet._conv_block(3, base_channels)
        self.encoder2 = SimpleUNet._conv_block(base_channels, base_channels * 2)
        self.encoder3 = SimpleUNet._conv_block(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.attn3 = SFABlock(base_channels * 4)
        self.attn2 = SFABlock(base_channels * 2)
        self.fuse2 = SimpleUNet._conv_block(base_channels * 6, base_channels * 2)
        self.fuse1 = SimpleUNet._conv_block(base_channels * 3, base_channels)
        self.classifier = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.attn3(self.encoder3(self.pool(e2)))
        up2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.fuse2(torch.cat([up2, self.attn2(e2)], dim=1))
        up1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.fuse1(torch.cat([up1, e1], dim=1))
        return {"out": self.classifier(d1)}


def build_model(model_name, num_classes, use_pretrained=True, aux_loss=False, architecture_params=None):
    _ensure_console_streams()
    _configure_offline_environment()

    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"不支持的模型类型: {model_name}")

    builder_fn, family, cls_in_channels = MODEL_BUILDERS[model_name]
    arch, backbone_name = split_model_name(model_name)
    arch_params = normalize_architecture_params(arch, architecture_params or {})

    if family == "unet":
        return SimpleUNet(num_classes=num_classes, base_channels=arch_params.get("base_channels", cls_in_channels)), family, cls_in_channels
    if family == "sfanet":
        return SFANet(num_classes=num_classes, base_channels=arch_params.get("base_channels", cls_in_channels)), family, cls_in_channels
    if family == "deeplabv3plus":
        return DeepLabV3PlusModel(
            backbone_name,
            num_classes=num_classes,
            use_pretrained=use_pretrained,
            config_params=arch_params
        ), family, cls_in_channels
    if family == "pspnet":
        return build_pspnet_resnet50(num_classes, use_pretrained), family, cls_in_channels
    if family == "upernet_swin":
        return build_upernet_swin_t(num_classes, use_pretrained), family, cls_in_channels

    weights = _resolve_seg_weights(model_name, use_pretrained)
    kwargs = {"weights": weights}
    builder_signature = inspect.signature(builder_fn).parameters
    if "progress" in builder_signature:
        kwargs["progress"] = False
    if "weights_backbone" in builder_signature:
        # Prevent torchvision from downloading backbone weights when full segmentation weights are unavailable.
        kwargs["weights_backbone"] = None
    requested_aux_loss = arch_params.get("aux_loss", aux_loss)
    if "aux_loss" in builder_signature:
        # torchvision segmentation builders require aux_loss=True when pretrained segmentation weights are used.
        if weights is not None and requested_aux_loss is False:
            logging.info(f"{model_name} 使用预训练分割权重时，aux_loss已自动从False调整为True")
            kwargs["aux_loss"] = True
        else:
            kwargs["aux_loss"] = requested_aux_loss
    base_model = _build_with_pretrained_fallback(builder_fn, model_name, {"aux_loss": requested_aux_loss}, **kwargs)
    replace_classifier_head(base_model, family, cls_in_channels, num_classes)
    return base_model, family, cls_in_channels


def detect_model_type_from_state_dict(state_dict):
    keys = list(state_dict.keys())
    is_unet = any(k.startswith("enc1.") for k in keys)
    is_sfanet = any(k.startswith("encoder1.") for k in keys) and any("attn3." in k for k in keys)
    is_pspnet = any(k.startswith("model.decoder.psp.") or ".decoder.psp." in k for k in keys)
    is_segformer = any("segformer.encoder.patch_embeddings" in k for k in keys)
    is_mask2former = any("model.model.pixel_level_module." in k or "model.pixel_level_module." in k for k in keys)
    is_segnext = any(k.startswith("stages.") for k in keys) and any(k.startswith("lateral_convs.") for k in keys)
    is_upernet_swin = any("upernet." in k or "decode_head.psp_modules" in k or "backbone.embeddings.patch_embeddings" in k for k in keys)
    is_deeplab_plus = any("low_level_proj." in k or "aspp.convs." in k for k in keys)
    is_deeplab = any("classifier.0.convs." in k for k in keys)
    is_fcn = any("classifier.4.weight" in k or "classifier.4.1.weight" in k for k in keys)

    if any("backbone.layer3.6." in k for k in keys):
        backbone = "ResNet101"
    elif any("backbone.layer" in k for k in keys):
        backbone = "ResNet50"
    elif any("backbone.low_features." in k for k in keys) or any("backbone.0." in k for k in keys):
        backbone = "MobileNetV3-Large"
    else:
        backbone = "ResNet50"

    if is_unet:
        model_name = "U-Net"
    elif is_sfanet:
        model_name = "SFA-Net"
    elif is_pspnet:
        model_name = compose_model_name("PSPNet", "ResNet50")
    elif is_segformer:
        model_name = compose_model_name("SegFormer", "B2")
    elif is_mask2former:
        model_name = compose_model_name("Mask2Former", "Swin-T")
    elif is_segnext:
        variant = "Small" if any("stages.2.3." in k or "lateral_convs.3.weight" in k and state_dict[k].shape[0] == 384 for k in keys) else "Tiny"
        model_name = compose_model_name("SegNeXt", variant)
    elif is_upernet_swin:
        model_name = compose_model_name("UPerNet-Swin", "Tiny")
    elif is_deeplab_plus:
        model_name = compose_model_name("DeepLabV3+", backbone)
    elif is_deeplab:
        model_name = compose_model_name("DeepLabV3", backbone)
    elif is_fcn:
        model_name = compose_model_name("FCN", backbone if backbone != "MobileNetV3-Large" else "ResNet50")
    else:
        model_name = get_default_model_name()

    if model_name not in MODEL_BUILDERS:
        logging.warning(f"自动识别到未注册模型类型 {model_name}，将回退默认模型")
        return get_default_model_name()
    return model_name
