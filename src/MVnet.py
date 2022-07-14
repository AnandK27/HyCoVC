import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

from src.helpers.endecoder import ME_Spynet, Warp_net, flow_warp
from src.network.mv import Analysis_mv_net
from src.network.mv import Synthesis_mv_net
from src.helpers import maths
from src.compression import mv_model

MIN_SCALE = 0.11
LOG_SCALES_MIN = -3.
MIN_LIKELIHOOD = 1e-9
MAX_LIKELIHOOD = 1e3

lower_bound_identity = maths.LowerBoundIdentity.apply
lower_bound_toward = maths.LowerBoundToward.apply

MVInfo = namedtuple(
    "MVInfo",
    "pred "
    "mv_z_nbpp mv_z_qbpp warpframe",
)


class CodingModel(nn.Module):
    """
    Probability model for estimation of (cross)-entropies in the context
    of data compression. TODO: Add tensor -> string compression and
    decompression functionality.
    """

    def __init__(self, min_likelihood=MIN_LIKELIHOOD, max_likelihood=MAX_LIKELIHOOD):
        super(CodingModel, self).__init__()
        self.min_likelihood = float(min_likelihood)
        self.max_likelihood = float(max_likelihood)

    def _quantize(self, x, mode='noise', means=None):
        """
        mode:       If 'noise', returns continuous relaxation of hard
                    quantization through additive uniform noise channel.
                    Otherwise perform actual quantization (through rounding).
        """

        if mode == 'noise':
            quantization_noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
            x = x + quantization_noise

        elif mode == 'quantize':
            if means is not None:
                x = x - means
                x = torch.floor(x + 0.5)
                x = x + means
            else:
                x = torch.floor(x + 0.5)
        else:
            raise NotImplementedError
        
        return x

    def _estimate_entropy(self, likelihood, spatial_shape):

        EPS = 1e-9  
        quotient = -np.log(2.)
        batch_size = likelihood.size()[0]

        assert len(spatial_shape) == 2, 'Mispecified spatial dims'
        n_pixels = np.prod(spatial_shape)

        log_likelihood = torch.log(likelihood + EPS)
        n_bits = torch.sum(log_likelihood) / (batch_size * quotient)
        bpp = n_bits / n_pixels

        return n_bits, bpp

    def _estimate_entropy_log(self, log_likelihood, spatial_shape):

        quotient = -np.log(2.)
        batch_size = log_likelihood.size()[0]

        assert len(spatial_shape) == 2, 'Mispecified spatial dims'
        n_pixels = np.prod(spatial_shape)

        n_bits = torch.sum(log_likelihood) / (batch_size * quotient)
        bpp = n_bits / n_pixels

        return n_bits, bpp

    def quantize_latents_st(self, inputs, means=None):
        # Latents rounded instead of additive uniform noise
        # Ignore rounding in backward pass
        values = inputs

        if means is not None:
            values = values - means

        delta = (torch.floor(values + 0.5) - values).detach()
        values = values + delta

        if means is not None:
            values = values + means

        return values

    def latent_likelihood(self, x, mean, scale):

        # Assumes 1 - CDF(x) = CDF(-x)
        x = x - mean
        x = torch.abs(x)
        cdf_upper = self.standardized_CDF((0.5 - x) / scale)
        cdf_lower = self.standardized_CDF(-(0.5 + x) / scale)

        # Naive
        # cdf_upper = self.standardized_CDF( (x + 0.5) / scale )
        # cdf_lower = self.standardized_CDF( (x - 0.5) / scale )

        likelihood_ = cdf_upper - cdf_lower
        likelihood_ = lower_bound_toward(likelihood_, self.min_likelihood)

        return likelihood_


class MVNet(CodingModel):
    def __init__(self, mv_channels = 128, likelihood_type='gaussian', entropy_code = True, vectorize_encoding = True, block_encode = True):
        super(MVNet, self).__init__()
        self.mv_channels = mv_channels
        self.opticFlow = ME_Spynet()
        self.mvEncoder = Analysis_mv_net(mv_channels)
        self.mvDecoder = Synthesis_mv_net(mv_channels)
        self.warpnet = Warp_net()

        self.mv_likelihood = mv_model.MVDensity(n_channels = mv_channels)

        if likelihood_type == 'gaussian':
            self.standardized_CDF = maths.standardized_CDF_gaussian
        elif likelihood_type == 'logistic':
            self.standardized_CDF = maths.standardized_CDF_logistic
        else:
            raise ValueError('Unknown likelihood model: {}'.format(likelihood_type))

        if entropy_code is True:
            print('Building prior probability tables...')
            self.MV_entropy_model = mv_model.MVEntropyModel(distribution=self.mv_likelihood)
            self.vectorize_encoding = vectorize_encoding
            self.block_encode = block_encode 



    def forward(self, x, ref):
        est_mv = self.opticFlow(x, ref)
        mv_z = self.mvEncoder(est_mv)
        batch_shape = x.size(0)

        noisy_mv_z = self._quantize(mv_z, mode='noise')
        noisy_mv_z_likelihood = self.mv_likelihood(noisy_mv_z)
        noisy_mv_z_bits, noisy_mv_z_bpp = self._estimate_entropy(
            noisy_mv_z_likelihood, spatial_shape=est_mv.size()[2:])

        # Discrete entropy, mv_z
        quantized_mv_z = self._quantize(mv_z, mode='quantize')
        quantized_mv_z_likelihood = self.mv_likelihood(quantized_mv_z)
        quantized_mv_z_bits, quantized_mv_z_bpp = self._estimate_entropy(
            quantized_mv_z_likelihood, spatial_shape=est_mv.size()[2:])

        if self.training is True:
            mv_z_decoded = noisy_mv_z
        else:
            mv_z_decoded = quantized_mv_z

        mv_upsample = self.mvDecoder(mv_z_decoded)

        prediction, warpframe = self.motioncompensation(ref, mv_upsample)

        info = MVInfo(
            pred=prediction,
            mv_z_nbpp=noisy_mv_z_bpp,
            mv_z_qbpp=quantized_mv_z_bpp,
            warpframe=warpframe
        )
        
        return info

    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe