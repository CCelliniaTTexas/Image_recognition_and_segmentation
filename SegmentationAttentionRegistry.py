import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from CGAM import CGAM


ATTENTION_TYPES = {
    "none": "不使用注意力",
    "cross": "交叉注意力",
    "cgam": "曲率几何注意力",
}

DEFAULT_ATTENTION_TYPE = "cross"


def get_attention_options():
    return list(ATTENTION_TYPES.values())


def get_default_attention_type():
    return ATTENTION_TYPES[DEFAULT_ATTENTION_TYPE]


def get_attention_display_name(attention_type):
    key = normalize_attention_type(attention_type)
    return ATTENTION_TYPES.get(key, ATTENTION_TYPES["none"])


def normalize_attention_type(attention_type):
    if attention_type in ATTENTION_TYPES:
        return attention_type
    for key, label in ATTENTION_TYPES.items():
        if attention_type == label:
            return key
    return "none"


def resolve_attention_in_channels(model_name, model, fallback_channels):
    """Resolve channels at the segmentation-head insertion point."""
    classifier = getattr(model, "classifier", None)

    if hasattr(classifier, "aspp"):
        return _get_module_out_channels(classifier.aspp) or int(fallback_channels)

    if isinstance(classifier, nn.Sequential):
        if model_name.startswith("DeepLabV3"):
            return _get_module_out_channels(classifier[0]) or int(fallback_channels)
        if model_name.startswith("FCN"):
            return _get_module_out_channels(classifier[0]) or int(fallback_channels)

    return int(fallback_channels)


class CrossAttention(nn.Module):
    """交叉注意力机制。"""

    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        num_heads = 8
        depth = 1
        dropout = 0.1

        self.depth = depth
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert self.head_dim * num_heads == in_channels, "in_channels必须能被num_heads整除"

        self.attentions = nn.ModuleList()
        self.layer_norms1 = nn.ModuleList()
        self.layer_norms2 = nn.ModuleList()
        self.ffns = nn.ModuleList()

        for _ in range(depth):
            self.attentions.append(nn.MultiheadAttention(
                embed_dim=in_channels,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ))
            self.layer_norms1.append(nn.LayerNorm(in_channels))
            self.layer_norms2.append(nn.LayerNorm(in_channels))
            self.ffns.append(nn.Sequential(
                nn.Linear(in_channels, in_channels * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(in_channels * 4, in_channels),
                nn.Dropout(dropout)
            ))

    def forward(self, query, key, value):
        """query/key/value: [B, C, H, W] 特征图。"""
        batch, channels, height, width = query.size()
        query = query.view(batch, channels, height * width).permute(0, 2, 1)

        key_batch, key_channels, key_height, key_width = key.size()
        key = key.view(key_batch, key_channels, key_height * key_width).permute(0, 2, 1)
        value = value.view(key_batch, key_channels, key_height * key_width).permute(0, 2, 1)

        for i in range(self.depth):
            residual = query
            attn_output, _ = self.attentions[i](query=query, key=key, value=value)
            query = self.layer_norms1[i](residual + attn_output)

            residual = query
            query = self.layer_norms2[i](residual + self.ffns[i](query))

        return query.permute(0, 2, 1).view(batch, channels, height, width)


class AttentionEnhancedModel(nn.Module):
    """交叉注意力机制的分割模型包装器。"""

    def __init__(self, base_model, in_channels, family=None):
        super(AttentionEnhancedModel, self).__init__()
        self.base_model = base_model
        self.family = family
        self.attention = CrossAttention(in_channels=in_channels)
        self.attention_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        feature_model = self.base_model
        if hasattr(self.base_model, "base_model") and hasattr(self.base_model.base_model, "backbone"):
            feature_model = self.base_model.base_model

        features = feature_model.backbone(x)
        if hasattr(self.base_model, "cgam"):
            def apply_stacked_attention(feat):
                feat = self.base_model.cgam(feat)
                attended = self.attention(feat, feat, feat)
                return feat + self.attention_scale * (attended - feat)

            logits = _run_classifier_with_attention(
                feature_model,
                features,
                x,
                apply_stacked_attention,
                self.family,
            )
        else:
            logits = _run_classifier_with_attention(
                feature_model,
                features,
                x,
                self._apply_attention_residual,
                self.family,
            )
        return {"out": logits}

    def _apply_attention_residual(self, feat):
        attended = self.attention(feat, feat, feat)
        return feat + self.attention_scale * (attended - feat)


class CGAMEnhancedModel(nn.Module):
    """曲率几何注意力 (CGAM) 的分割模型包装器。"""

    def __init__(self, base_model, in_channels, reduction=None, family=None):
        super(CGAMEnhancedModel, self).__init__()
        self.base_model = base_model
        self.family = family

        high_reduction = reduction or self._choose_reduction(in_channels)
        self.cgam = CGAM(in_channels=in_channels, reduction=high_reduction, residual_init=0.08)

        self.low_cgam = None
        classifier = getattr(base_model, "classifier", None)
        if family == "deeplabv3plus" and hasattr(classifier, "low_level_proj"):
            low_channels = _get_module_out_channels(classifier.low_level_proj)
            if low_channels is not None:
                self.low_cgam = CGAM(
                    in_channels=low_channels,
                    reduction=self._choose_reduction(low_channels),
                    residual_init=0.12,
                )
                logging.info(
                    f"DeepLabV3+ 已启用双层CGAM: low={low_channels}通道, high={in_channels}通道"
                )

    @staticmethod
    def _choose_reduction(in_channels):
        if in_channels <= 64:
            return 2
        if in_channels <= 256:
            return 4
        return 8

    def forward(self, x):
        features = self.base_model.backbone(x)
        if self.low_cgam is not None:
            logits = self._forward_deeplabv3plus_dual_cgam(features, x)
        else:
            logits = _run_classifier_with_attention(self.base_model, features, x, self.cgam, self.family)
        return {"out": logits}

    def _forward_deeplabv3plus_dual_cgam(self, features, input_tensor):
        classifier = self.base_model.classifier
        low = classifier.low_level_proj(features["low_level"])
        low = self.low_cgam(low)
        high = self.cgam(classifier.aspp(features["out"]))
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        logits = classifier.classifier(classifier.decoder(torch.cat([low, high], dim=1)))
        return _resize_logits(logits, input_tensor)


def _run_classifier(base_model, features, input_tensor):
    try:
        logits = base_model.classifier(features)
    except Exception:
        logits = base_model.classifier(features["out"])

    if logits.shape[-2:] != input_tensor.shape[-2:]:
        logits = F.interpolate(logits, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def _run_classifier_with_attention(base_model, features, input_tensor, attention_fn, family=None):
    classifier = getattr(base_model, "classifier", None)

    if hasattr(classifier, "aspp") and hasattr(classifier, "decoder") and hasattr(classifier, "low_level_proj"):
        low = classifier.low_level_proj(features["low_level"])
        high = attention_fn(classifier.aspp(features["out"]))
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        logits = classifier.classifier(classifier.decoder(torch.cat([low, high], dim=1)))
        return _resize_logits(logits, input_tensor)

    if isinstance(classifier, nn.Sequential):
        x = features["out"] if isinstance(features, dict) else features
        insert_after = _get_sequential_insert_index(classifier, family)
        for idx, module in enumerate(classifier):
            x = module(x)
            if idx == insert_after:
                x = attention_fn(x)
        return _resize_logits(x, input_tensor)

    return _run_classifier(base_model, features, input_tensor)


def _resize_logits(logits, input_tensor):
    if logits.shape[-2:] != input_tensor.shape[-2:]:
        logits = F.interpolate(logits, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def _get_sequential_insert_index(classifier, family):
    if family == "deeplabv3" or classifier[0].__class__.__name__.lower() == "aspp":
        return 0
    if family == "fcn":
        return min(3, len(classifier) - 2)
    return max(0, len(classifier) - 2)


def _get_module_out_channels(module):
    if hasattr(module, "out_channels"):
        return int(module.out_channels)
    for sub_module in reversed(list(module.modules())):
        if hasattr(sub_module, "out_channels"):
            return int(sub_module.out_channels)
    return None


def apply_attention_to_model(model, model_name, family, feature_dim, attention_type):
    attention_key = normalize_attention_type(attention_type)
    if attention_key == "none":
        return model
    attention_channels = resolve_attention_in_channels(model_name, model, feature_dim)

    if attention_key == "cross":
        if family in ("deeplabv3", "deeplabv3plus", "fcn"):
            return AttentionEnhancedModel(model, in_channels=attention_channels, family=family)
        logging.warning(f"{model_name} 当前不支持交叉注意力封装，已自动忽略该选项")
        return model

    if attention_key == "cgam":
        supported_families = ("deeplabv3", "deeplabv3plus", "fcn")
        if family in supported_families and hasattr(model, "backbone") and hasattr(model, "classifier"):
            return CGAMEnhancedModel(model, in_channels=attention_channels, family=family)
        logging.warning(f"{model_name} 当前不支持CGAM注入，已自动忽略该选项")
        return model

    return model
