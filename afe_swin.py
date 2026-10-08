import torch
from torch import nn
import timm


class AdaptiveFrequencyEnhancement(nn.Module):
    def __init__(self, dim, h=12, w=12):
        super().__init__()
        self.h, self.w = h, w
        weight = torch.zeros(dim, h, w // 2 + 1, 2)
        weight[..., 0] = 1
        self.complex_weight = nn.Parameter(weight)
        self.alpha = nn.Parameter(torch.zeros(1))

    def components(self, f_spatial):
        if f_spatial.ndim != 4:
            raise ValueError("F_spatial must be BCHW")
        _, channels, height, width = f_spatial.shape
        if (channels, height, width) != self.complex_weight.shape[:2] + (self.w,):
            raise ValueError(f"Expected C,H,W={self.complex_weight.shape[0]},{self.h},{self.w}; "
                             f"got {channels},{height},{width}")
        work = f_spatial.float() if f_spatial.dtype in (torch.float16, torch.bfloat16) else f_spatial

        with torch.autocast(device_type=work.device.type, enabled=False):
            f_freq = torch.fft.rfft2(work, dim=(-2, -1), norm="ortho")
            w_complex = torch.view_as_complex(self.complex_weight.contiguous())
            delta_f = torch.fft.irfft2(f_freq * (w_complex - 1), s=(height, width),
                                      dim=(-2, -1), norm="ortho")
        f_enhance = work + delta_f
        z_s = work.mean(dim=(-2, -1))
        z_f = f_enhance.mean(dim=(-2, -1))
        z = z_s + self.alpha * (z_f - z_s)
        return {"f_spatial": work, "f_freq": f_freq, "w_complex": w_complex,
                "delta_f": delta_f, "f_enhance": f_enhance, "z_s": z_s, "z_f": z_f, "z": z,
                "spatial": work, "spectrum": f_freq, "weight": w_complex,
                "delta": delta_f, "enhanced": f_enhance}

    def forward(self, f_spatial):
        return self.components(f_spatial)["z"]


class AFE_Swin(nn.Module):

    def __init__(self, num_classes=8, pretrained=True, config=None):
        super().__init__()
        if config is None:
            config = {"model_name": "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
                      "feature_hw": 12, "head_dropout": 0.2}
        self.backbone = timm.create_model(config["model_name"], pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad = "layers.2" in name or "layers.3" in name or "norm" in name

        self.freq_prior_module = AdaptiveFrequencyEnhancement(self.num_features,
                                                              config["feature_hw"], config["feature_hw"])
        self.head = nn.Sequential(nn.Linear(self.num_features, self.num_features // 2),
                                  nn.LayerNorm(self.num_features // 2), nn.GELU(),
                                  nn.Dropout(config.get("head_dropout", 0.2)),
                                  nn.Linear(self.num_features // 2, num_classes))

    def _to_bchw(self, features):
        if features.ndim == 4 and features.shape[-1] == self.num_features:
            return features.permute(0, 3, 1, 2).contiguous()
        if features.ndim == 4 and features.shape[1] == self.num_features:
            return features.contiguous()
        if features.ndim == 3 and features.shape[-1] == self.num_features:
            batch, length, channels = features.shape
            height = int(length ** 0.5)
            if height * height == length:
                return features.transpose(1, 2).reshape(batch, channels, height, height).contiguous()
        raise ValueError(f"Unexpected Swin high-level feature shape: {tuple(features.shape)}")

    def extract_spatial_features(self, image):
        return self._to_bchw(self.backbone.forward_features(image))

    def forward_components(self, image):
        parts = self.freq_prior_module.components(self.extract_spatial_features(image))
        parts["logits"] = self.head(parts["z"])
        return parts

    def forward(self, image):
        return self.forward_components(image)["logits"]


AdaptiveFrequencyPriorBlock = AdaptiveFrequencyEnhancement
FSD_Swin = AFE_Swin
FPE_Swin = AFE_Swin
