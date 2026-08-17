"""
adnet_model.py

Skeleton PyTorch implementation of ADNET (Adaptive Dual-stream NETwork), matching
the architecture and equations described in Section 4 of the manuscript:

  "ADNET: Adaptive Dual-Stream Network with Hierarchical Attention Fusion for
   Early-Stage Alzheimer's Disease Detection from Brain MRI"

STATUS: this file was written during manuscript revision to satisfy the code-
availability requirement, using the equations and module descriptions already
in the paper. It has NOT been verified against the authors' original training
code. In particular:

  - Section 4.1.1 states Stream A (EfficientNet-B3 as configured in ADNET)
    contributes only 1.8M parameters, far below a standard EfficientNet-B3
    (~10.7M). This skeleton defaults to loading the FULL pretrained backbone
    from timm, which will NOT match that parameter count. Authors: replace
    `_build_stream_a()` with your actual (evidently truncated/width-reduced)
    backbone configuration, or update the manuscript to reflect a full-depth
    backbone if that's what was actually used.
  - The same caveat applies to Stream B (Swin-T, stated as 1.1M vs. a
    standard ~28M).
  - Equation numbers in comments refer to the manuscript's Section 4;
    see the manuscript's editorial note in that section for how they were
    reconstructed (the original submission had equation numbers but the
    formulas themselves were missing).

Once corrected, delete this status note.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None
    print("Warning: timm not installed. Install with `pip install timm` to use pretrained backbones.")


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation module -- Eq. (6)
#   z = GAP(F) in R^C
#   s = sigma(W2 . delta(W1 . z))
#   F~ = F (x) s
# ---------------------------------------------------------------------------
class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        z = self.gap(x).view(b, c)                 # Eq. 6: z = GAP(F)
        s = self.relu(self.fc1(z))
        s = self.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s                                # Eq. 6: F~ = F (x) s


# ---------------------------------------------------------------------------
# Channel-Spatial Progressive Attention (CSPA) -- Eq. (12)-(17)
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """Eq. (12)-(15): channel attention (CBAM-style)."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)           # Eq. 12: F_avg = GAP(F)
        mx = F.adaptive_max_pool2d(x, 1)             # Eq. 13: F_max = GMP(F)
        mc = self.sigmoid(self.mlp(avg) + self.mlp(mx))   # Eq. 14-15
        return x * mc                                # F' = M_c (x) F


class SpatialAttention(nn.Module):
    """Eq. (16): spatial attention via channel-pooled 7x7 conv."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        ms = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # Eq. 16
        return ms


class CSPA(nn.Module):
    """
    Eq. (17): F_CSPA = F' (x) (M_s . M_AP)

    M_AP is an anatomical-prior mask (Section 4.2) -- a learnable or fixed
    spatial prior initialized toward hippocampal/entorhinal regions. This
    skeleton uses a learnable prior initialized as a soft Gaussian blob at
    the map center as a placeholder; replace with the authors' actual
    anatomical-prior initialization scheme (Section 4.2 describes this
    conceptually but the manuscript does not give exact initialization
    parameters -- authors should fill this in).
    """

    def __init__(self, channels, spatial_size=7, reduction=8):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()
        # Placeholder anatomical prior -- see docstring above.
        self.anatomical_prior = nn.Parameter(torch.ones(1, 1, spatial_size, spatial_size) * 0.5)

    def forward(self, x):
        f_prime = self.channel_attn(x)               # Stage 1: channel attention
        m_s = self.spatial_attn(f_prime)              # Stage 2: spatial attention, Eq. 16
        m_ap = torch.sigmoid(
            F.interpolate(self.anatomical_prior, size=x.shape[-2:], mode="bilinear", align_corners=False)
        )
        f_cspa = f_prime * (m_s * m_ap)                # Eq. 17
        return f_cspa


# ---------------------------------------------------------------------------
# Hierarchical Attention Fusion (HAF) -- Eq. (18)-(21)
# ---------------------------------------------------------------------------
class HAF(nn.Module):
    """
    Three-scale progressive fusion of Stream A and Stream B CSPA outputs.
    Eq. (18): H1 = Conv1x1([F_CSPA_A^1 ; F_CSPA_B^1])
    Eq. (19): H2 = Conv1x1([F_CSPA_A^2 ; F_CSPA_B^2 ; Up(H1)])
    Eq. (20): H3 = Conv1x1([F_CSPA_A^3 ; F_CSPA_B^3 ; Up(H2)])
    Eq. (21): v  = GAP(H3)
    """

    def __init__(self, ch_a1, ch_b1, ch_a2, ch_b2, ch_a3, ch_b3, out_dim=256):
        super().__init__()
        self.fuse1 = nn.Conv2d(ch_a1 + ch_b1, out_dim, kernel_size=1)
        self.fuse2 = nn.Conv2d(ch_a2 + ch_b2 + out_dim, out_dim, kernel_size=1)
        self.fuse3 = nn.Conv2d(ch_a3 + ch_b3 + out_dim, out_dim, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, fa1, fb1, fa2, fb2, fa3, fb3):
        h1 = self.fuse1(torch.cat([fa1, fb1], dim=1))                                  # Eq. 18
        h1_up = F.interpolate(h1, size=fa2.shape[-2:], mode="bilinear", align_corners=False)
        h2 = self.fuse2(torch.cat([fa2, fb2, h1_up], dim=1))                           # Eq. 19
        h2_up = F.interpolate(h2, size=fa3.shape[-2:], mode="bilinear", align_corners=False)
        h3 = self.fuse3(torch.cat([fa3, fb3, h2_up], dim=1))                           # Eq. 20
        v = self.gap(h3).flatten(1)                                                     # Eq. 21
        return v


# ---------------------------------------------------------------------------
# Classification head -- Eq. (22)-(23)
# ---------------------------------------------------------------------------
class ClassificationHead(nn.Module):
    """
    Eq. (22): h = Dropout(delta(W_fc1 . v + b1))
    Eq. (23): p = softmax(W_fc2 . h + b2)
    """

    def __init__(self, in_dim, hidden_dim=128, num_classes=4, dropout=0.4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, v):
        h = self.dropout(self.relu(self.fc1(v)))     # Eq. 22
        logits = self.fc2(h)                          # Eq. 23 (softmax applied in loss/inference)
        return logits


# ---------------------------------------------------------------------------
# Full ADNET model
# ---------------------------------------------------------------------------
class ADNET(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, cspa_reduction=8):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required: pip install timm")

        # --- Stream A: EfficientNet-B3 (CNN) ---
        # CAUTION: see module docstring -- full-depth EfficientNet-B3 has ~10.7M
        # params, not the 1.8M stated in the manuscript. Confirm/replace this.
        self.stream_a = timm.create_model(
            "efficientnet_b3", pretrained=pretrained, features_only=True,
            out_indices=(2, 3, 4),  # three intermediate scales, matching Eq. context (F_A1, F_A2, F_A3)
        )
        a_channels = self.stream_a.feature_info.channels()

        # --- Stream B: Swin-Tiny (Transformer) ---
        # CAUTION: see module docstring -- full-depth Swin-T has ~28M params,
        # not the 1.1M stated in the manuscript. Confirm/replace this.
        #
        # IMPORTANT GOTCHA: timm's Swin Transformer, unlike its CNN backbones,
        # naturally represents features channel-last (B, H, W, C) internally,
        # and depending on the installed timm version, features_only=True may
        # return that format instead of the (B, C, H, W) the rest of this model
        # assumes. We request NCHW explicitly where supported, and additionally
        # auto-detect and fix the format at runtime in forward() below as a
        # safety net -- silently training on transposed feature maps would not
        # crash, it would just quietly produce a broken model.
        try:
            self.stream_b = timm.create_model(
                "swin_tiny_patch4_window7_224", pretrained=pretrained, features_only=True,
                out_indices=(1, 2, 3), output_fmt="NCHW",
            )
        except TypeError:
            # older timm versions don't accept output_fmt; fall back and rely
            # on the runtime auto-detect/permute in forward()
            self.stream_b = timm.create_model(
                "swin_tiny_patch4_window7_224", pretrained=pretrained, features_only=True,
                out_indices=(1, 2, 3),
            )
        b_channels = self.stream_b.feature_info.channels()

        # --- SE modules applied within Stream A's MBConv blocks (Eq. 6) ---
        # timm's efficientnet already includes SE internally; this is exposed
        # here only for cases where a custom backbone without built-in SE is used.
        self.se_extra = SqueezeExcitation(a_channels[-1], reduction=4)

        # --- CSPA applied to each stream's final-scale features (Eq. 12-17) ---
        self.cspa_a = CSPA(a_channels[-1], reduction=cspa_reduction)
        self.cspa_b = CSPA(b_channels[-1], reduction=cspa_reduction)

        # --- HAF: three-scale fusion (Eq. 18-21) ---
        # channel order matches forward()'s deepest-first call convention:
        # a_channels/b_channels from timm are ordered shallow->deep, so we
        # reverse here (index 2 = deepest = HAF's first input).
        self.haf = HAF(
            ch_a1=a_channels[2], ch_b1=b_channels[2],   # deepest (coarsest)
            ch_a2=a_channels[1], ch_b2=b_channels[1],   # mid
            ch_a3=a_channels[0], ch_b3=b_channels[0],   # shallowest (finest)
            out_dim=256,
        )

        # --- Classification head (Eq. 22-23) ---
        self.head = ClassificationHead(in_dim=256, num_classes=num_classes)

    @staticmethod
    def _ensure_nchw(feat):
        """
        Defensive fix for timm Swin Transformer feature maps sometimes being
        returned as (B, H, W, C) instead of (B, C, H, W). Heuristic: a 4D
        conv-style feature map should have its smallest dimension be the
        channel dimension in typical CNN feature maps (channels << H, W is
        NOT reliable in general, but for this specific gotcha the tell is
        that H and W, dims 1 and 2, will be equal to each other in NHWC
        format while dim 1 (would-be channel) is not, whereas in NCHW dims
        2 and 3 (H, W) are equal). We check both plausible signatures and
        permute only if the tensor is confidently NHWC.
        """
        if feat.dim() != 4:
            return feat
        b, d1, d2, d3 = feat.shape
        looks_nhwc = (d1 == d2) and (d1 != d3)
        looks_nchw = (d2 == d3) and (d2 != d1)
        if looks_nhwc and not looks_nchw:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, x):
        # timm's features_only with out_indices=(2,3,4) returns features ordered
        # shallow -> deep, i.e. f_shallow has the LARGEST spatial size and
        # f_deep the SMALLEST. HAF's Eq. (18)-(20) upsamples H1 to match the
        # next scale, which only makes sense if H1 starts as the SMALLEST
        # (deepest) map and progressively fuses toward the LARGEST (shallowest)
        # -- i.e. standard coarse-to-fine (FPN-style) fusion. We therefore feed
        # HAF deepest-first here. VERIFY this ordering against your actual
        # training code -- the manuscript's prose does not unambiguously state
        # which direction "scale 1/2/3" refers to, and getting this backwards
        # will silently produce a working-but-wrong model.
        f_shallow_a, f_mid_a, f_deep_a = self.stream_a(x)
        f_shallow_b, f_mid_b, f_deep_b = self.stream_b(x)

        # Safety net for the NHWC-vs-NCHW gotcha described above: if a Stream B
        # feature map's channel dimension does not appear where expected (dim=1)
        # but does appear last instead, permute it back to NCHW before use.
        f_shallow_b, f_mid_b, f_deep_b = (
            self._ensure_nchw(f_shallow_b), self._ensure_nchw(f_mid_b), self._ensure_nchw(f_deep_b)
        )

        f_deep_a = self.cspa_a(f_deep_a)   # Eq. 17, applied to Stream A's deepest scale
        f_deep_b = self.cspa_b(f_deep_b)   # Eq. 17, applied to Stream B's deepest scale

        v = self.haf(
            f_deep_a, f_deep_b,      # Eq. 18: "scale 1" = coarsest
            f_mid_a, f_mid_b,        # Eq. 19: "scale 2" = mid
            f_shallow_a, f_shallow_b,  # Eq. 20: "scale 3" = finest
        )
        logits = self.head(v)                          # Eq. 22-23
        return logits


# ---------------------------------------------------------------------------
# Composite loss: weighted cross-entropy + focal loss -- Eq. (24)-(26)
# ---------------------------------------------------------------------------
class CompositeLoss(nn.Module):
    """
    Eq. (24): L_WCE = -sum_c w_c * y_c^LS * log(p_c)
    Eq. (25): L_FL  = -sum_c alpha_c * (1 - p_c)^gamma * y_c * log(p_c)
    Eq. (26): L_total = L_WCE + lambda * L_FL
    """

    def __init__(self, class_weights, alpha=0.25, gamma=2.0, lam=0.3, label_smoothing=0.1):
        super().__init__()
        self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        y_onehot = F.one_hot(targets, num_classes).float()
        y_ls = y_onehot * (1 - self.label_smoothing) + self.label_smoothing / num_classes  # Eq. 24 (y_c^LS)

        wce = -(self.class_weights.unsqueeze(0) * y_ls * log_probs).sum(dim=1).mean()       # Eq. 24

        pt = (probs * y_onehot).sum(dim=1)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        fl = -(focal_weight * (y_onehot * log_probs).sum(dim=1)).mean()                     # Eq. 25

        return wce + self.lam * fl                                                          # Eq. 26


if __name__ == "__main__":
    # Smoke test
    model = ADNET(num_classes=4, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect [2, 4]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params / 1e6:.2f}M")
    print("NOTE: this will be far larger than the manuscript's stated 3.2M")
    print("until Stream A/B are replaced with the actual (truncated) backbones used.")
