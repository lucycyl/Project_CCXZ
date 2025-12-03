"""
Learnable 1D wavelet decomposition using NeuralDWAV, drop-in for WPMixer.

API is matched to the original Decomposition in DWT_Decomposition.py:
    - same __init__ signature
    - transform(x) / inv_transform(...)
    - input_w_dim / pred_w_dim shapes

x is always [batch, channel, seq].
"""

import torch
import torch.nn as nn

from NeuralDWAV import NeuralDWAV


class Decomposition(nn.Module):
    def __init__(self,
                 input_length=[],
                 pred_length=[],
                 wavelet_name=[],
                 level=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 tfactor=[],
                 dfactor=[],
                 device=[],
                 no_decomposition=[],
                 use_amp=[]):
        super(Decomposition, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.device = device
        self.no_decomposition = no_decomposition
        self.use_amp = use_amp
        self.eps = 1e-5

        self.tfactor = tfactor
        self.dfactor = dfactor

        # We keep the "no_decomposition" logic identical to the original.
        if not self.no_decomposition:
            # Use NeuralDWAV in DWT mode as a learnable analogue of the original DWT.
            #   - Input_Size: sequence length
            #   - Input_Level: same "level" as before
            #   - Input_Archi="DWT": produces level+1 bands like the old DWT,
            #                        so we don't have to touch WPMixer's branches.
            #
            # Filt_Mother is only used for initialization; after that, filters are trainable.
            mother = self.wavelet_name if isinstance(self.wavelet_name, str) else "db4"
            self.ndwav = NeuralDWAV(
                Input_Size=self.input_length,
                Input_Level=self.level,
                Input_Archi="DWT",          # keep topology compatible with old Decomposition
                Filt_Trans=True,
                Filt_Train=True,
                Filt_Tfree=False,
                Filt_Style="Filter_Free",
                Filt_Mother=mother,
                Act_Train=True,
                Act_Style="Sigmoid",        # or whatever default you want
                Act_Symmetric=True,
                Act_Init=0
            ).to(self.device)

            # Infer coefficient lengths by doing a dummy forward,
            # exactly like the original Decomposition did with DWT1DForward. :contentReference[oaicite:1]{index=1}
            self.input_w_dim = self._dummy_forward(self.input_length)
            self.pred_w_dim = self._scale_pred_dims(self.input_w_dim,
                                                    self.input_length,
                                                    self.pred_length)
        else:
            # No decomposition: single branch with full length
            self.input_w_dim = [self.input_length]
            self.pred_w_dim = [self.pred_length]

        #################################
        # Affine branch kept for compatibility (off by default)
        #################################
        self.affine = False
        if self.affine and not self.no_decomposition:
            self._init_params()

    # ------------------------------------------------------------------
    # Public API – same as original Decomposition
    # ------------------------------------------------------------------
    def transform(self, x):
        """
        x: [batch, channel, seq]
        Returns:
            yl: [batch, channel, L0]      (approximation / low-pass)
            yh: list of length `level`,
                each yh[i]: [batch, channel, Li] (detail bands)
        """
        if not self.no_decomposition:
            yl, yh = self._wavelet_decompose(x)
        else:
            yl, yh = x, []  # identity
        return yl, yh

    def inv_transform(self, yl, yh):
        """
        yl, yh in the same format as returned by transform().
        Returns:
            x: [batch, channel, seq_reconstructed]
        """
        if not self.no_decomposition:
            x = self._wavelet_reverse_decompose(yl, yh)
        else:
            x = yl
        return x

    # ------------------------------------------------------------------
    # Dummy forward to infer band lengths (like original _dummy_forward)
    # ------------------------------------------------------------------
    def _dummy_forward(self, input_length):
        """
        Compute coefficient lengths by running NeuralDWAV.LDWT
        on a dummy tensor, then extracting the shapes.
        """
        dummy_x = torch.ones(
            (self.batch_size, self.channel, input_length), device=self.device
        )
        B, C, L = dummy_x.shape

        # NeuralDWAV expects [B', 1, L]; we share filters across channels,
        # so flatten (B, C) into batch dimension.
        x = dummy_x.view(B * C, 1, L)

        embeddings = self.ndwav.LDWT(x)  # list of length level+1

        # embeddings[i] is the detail at level i+1, embeddings[level] is A_L.
        l = []
        # low-pass/coarsest approx
        l.append(embeddings[self.level].shape[-1])
        # detail bands, in order matching the original DWT_Decomposition logic:
        # [yh[0], yh[1], ..., yh[level-1]] :contentReference[oaicite:2]{index=2}
        for i in range(self.level):
            l.append(embeddings[i].shape[-1])
        return l

    def _scale_pred_dims(self, input_w_dim, input_length, pred_length):
        """
        Original code computed pred_w_dim via a dummy DWT on a dummy pred_length
        input. To keep things simple and compatible, we scale lengths linearly.
        """
        ratio = pred_length / float(input_length)
        pred_w_dim = [max(1, int(round(ratio * L))) for L in input_w_dim]
        return pred_w_dim

    # ------------------------------------------------------------------
    # Optional affine parameters (kept from original, off by default)
    # ------------------------------------------------------------------
    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones((self.level + 1, self.channel)))
        self.affine_bias = nn.Parameter(torch.zeros((self.level + 1, self.channel)))

    def _apply_affine(self, yl, yh):
        # yl: [B, C, L0], yh: list of [B, C, Li]
        yl_ = yl.transpose(1, 2)  # [B, L0, C]
        yl_ = yl_ * self.affine_weight[0] + self.affine_bias[0]
        yl = yl_.transpose(1, 2)  # [B, C, L0]

        for i in range(self.level):
            yh_i = yh[i].transpose(1, 2)  # [B, Li, C]
            yh_i = yh_i * self.affine_weight[i + 1] + self.affine_bias[i + 1]
            yh[i] = yh_i.transpose(1, 2)  # [B, C, Li]

        return yl, yh

    def _undo_affine(self, yl, yh):
        yl_ = yl.transpose(1, 2)
        yl_ = (yl_ - self.affine_bias[0]) / (self.affine_weight[0] + self.eps)
        yl = yl_.transpose(1, 2)

        for i in range(self.level):
            yh_i = yh[i].transpose(1, 2)
            yh_i = (yh_i - self.affine_bias[i + 1]) / (self.affine_weight[i + 1] + self.eps)
            yh[i] = yh_i.transpose(1, 2)

        return yl, yh

    # ------------------------------------------------------------------
    # Internal: forward / inverse via NeuralDWAV
    # ------------------------------------------------------------------
    def _wavelet_decompose(self, x):
        """
        x: [B, C, T]
        returns:
            yl: [B, C, L0]
            yh: list of [B, C, Li] (len == level)
        """
        B, C, T = x.shape

        # Flatten channels into batch, as NeuralDWAV is single-channel.
        x_flat = x.view(B * C, 1, T)

        embeddings = self.ndwav.LDWT(x_flat)
        # embeddings[0..level-1]: detail bands
        # embeddings[level]: approximation band

        yl = embeddings[self.level].view(B, C, -1)
        yh = []
        for i in range(self.level):
            yh_i = embeddings[i].view(B, C, -1)
            yh.append(yh_i)

        if self.affine:
            yl, yh = self._apply_affine(yl, yh)

        return yl, yh

    def _wavelet_reverse_decompose(self, yl, yh):
        """
        yl: [B, C, L0]
        yh: list of [B, C, Li]
        returns:
            x: [B, C, T]
        """
        B, C, _ = yl.shape

        if self.affine:
            yl, yh = self._undo_affine(yl, yh)

        # Build embeddings list in NeuralDWAV's expected order:
        #   embeddings[0..level-1] = detail bands (same order as in LDWT),
        #   embeddings[level]      = approximation band.
        embeddings = [None] * (self.level + 1)
        for i in range(self.level):
            embeddings[i] = yh[i].contiguous().view(B * C, 1, -1)
        embeddings[self.level] = yl.contiguous().view(B * C, 1, -1)

        x_flat = self.ndwav.iLDWT(embeddings)     # [B*C, 1, T]
        x = x_flat.view(B, C, -1)                 # [B, C, T]
        return x