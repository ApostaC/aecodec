import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
#from subnet import *
from .subnet import *
import torchac

DEBUG_USE_MPEG = False

def save_model(model, iter):
    torch.save(model.state_dict(), "./snapshot/iter{}.model".format(iter))

def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter') != -1 and f.find('.model') != -1:
        st = f.find('iter') + 4
        ed = f.find('.model', st)
        return int(f[st:ed])
    else:
        return 0



class VideoCompressor(nn.Module):
    def __init__(self):
        super(VideoCompressor, self).__init__()
        # self.imageCompressor = ImageCompressor()
        self.opticFlow = ME_Spynet()
        self.mvEncoder = Analysis_mv_net()
        self.Q = None
        self.mvDecoder = Synthesis_mv_net()
        self.warpnet = Warp_net()
        self.resEncoder = Analysis_net()
        self.resDecoder = Synthesis_net()
        self.respriorEncoder = Analysis_prior_net()
        self.respriorDecoder = Synthesis_prior_net()
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.bitEstimator_mv = BitEstimator(out_channel_mv)
        # self.flow_warp = Resample2d()
        # self.bitEstimator_feature = BitEstimator(out_channel_M)
        self.warp_weight = 0
        self.mxrange = 150
        self.calrealbits = False

        self.quantization_param = 1

    def set_quantization_param(self, q):
        self.quantization_param = q

    def forwardFirstFrame(self, x):
        output, bittrans = self.imageCompressor(x)
        cost = self.bitEstimator(bittrans)
        return output, cost

    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe

    def forward(self, input_image, referframe, quant_noise_feature=None, quant_noise_z=None, quant_noise_mv=None):
        estmv = self.opticFlow(input_image, referframe)
        mvfeature = self.mvEncoder(estmv)
        if self.training:
            quant_mv = mvfeature + quant_noise_mv
        else:
            quant_mv = torch.round(mvfeature)
            #quant_mv = torch.zeros(quant_mv.size(), device=quant_mv.device)
        quant_mv_upsample = self.mvDecoder(quant_mv)

        prediction, warpframe = self.motioncompensation(referframe, quant_mv_upsample)

        input_residual = input_image - prediction

        feature = self.resEncoder(input_residual)
        batch_size = feature.size()[0]
        z = self.respriorEncoder(feature)

        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)

        recon_sigma = self.respriorDecoder(compressed_z)

        feature_renorm = feature

        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
            #compressed_feature_renorm = torch.zeros(feature_renorm.size(), device=feature_renorm.device)

        recon_res = self.resDecoder(compressed_feature_renorm)
        recon_image = prediction + recon_res

        clipped_recon_image = recon_image.clamp(0., 1.)


        # distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        # psnr = tf.cond(
        #     tf.equal(mse_loss, 0), lambda: tf.constant(100, dtype=tf.float32),
        #     lambda: 10 * (tf.log(1 * 1 / mse_loss) / np.log(10)))

        warploss = torch.mean((warpframe - input_image).pow(2))
        interloss = torch.mean((prediction - input_image).pow(2))
        

        # bit per pixel

        def feature_probs_based_sigma(feature, sigma):
            
            def getrealbitsg(x, gaussian):
                # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(gaussian.cdf(i - 0.5).view(n,c,h,w,1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()

                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                return sym_out - self.mxrange, real_bits


            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-5, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
            
            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbitsg(feature, gaussian)
                total_bits = real_bits

            return total_bits, probs

        def iclr18_estrate_bits_z(z):
            
            def getrealbits(x):
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                return sym_out - self.mxrange, real_bits

            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbits(z)
                total_bits = real_bits

            return total_bits, prob


        def iclr18_estrate_bits_mv(mv):

            def getrealbits(x):
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
                return sym_out - self.mxrange, real_bits

            prob = self.bitEstimator_mv(mv + 0.5) - self.bitEstimator_mv(mv - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbits(mv)
                total_bits = real_bits

            return total_bits, prob

        total_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
        # entropy_context = entropy_context_from_sigma(compressed_feature_renorm, recon_sigma)
        total_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
        total_bits_mv, _ = iclr18_estrate_bits_mv(quant_mv)

        im_shape = input_image.size()

        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp_mv = total_bits_mv / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z + bpp_mv

        #tot_elements = torch.numel(compressed_feature_renorm) + torch.numel(compressed_z) + torch.numel(quant_mv)
        #print("Residual:", torch.unique(compressed_feature_renorm, return_counts=True))
        #print("MV:", torch.unique(quant_mv, return_counts=True))
        #print("Z:", torch.unique(compressed_z, return_counts=True))
        #print("Residual:", torch.sum(torch.abs(compressed_feature_renorm)), "out of", torch.numel(compressed_feature_renorm), compressed_feature_renorm.size())
        #print("MV:", torch.sum(torch.abs(quant_mv)), "out of", torch.numel(quant_mv), quant_mv.size())
        #print("Z:", torch.sum(torch.abs(compressed_z)), "out of", torch.numel(compressed_z), compressed_z.size())
        #print(quant_mv)
        #import pdb
        #pdb.set_trace()
        #elesize = 8 # XXX bits per element
        #bpp = tot_elements * elesize / (batch_size * im_shape[2] * im_shape[3])
        #bpp = torch.tensor(bpp)
        return clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp
        
    def round_values(self, v):
        #if self.quantization_param == 10:
        #    signs = torch.sign(v)
        #    v = torch.abs(v)
        #    v1 = (v <= 5).to(torch.int8) * v 
        #    v1 = torch.round(v1)
        #    v2 = (v > 5).to(torch.int8) * ((v - 5) / 10 + 5)
        #    v2 = torch.round(v2)
        #    return signs * (v1 + v2)
        #else: 
        #    return torch.round(v / self.quantization_param)

        #v = torch.round(v)
        #return torch.round(torch.sign(v) * torch.pow(torch.abs(v), 1/self.quantization_param))

        global DEBUG_USE_MPEG
        ''' for mpeg compression '''
        if DEBUG_USE_MPEG:
            return torch.round(v)

        ''' normal solution '''
        return torch.round(v / self.quantization_param) #* self.quantization_param


    def unround_values(self, v):
        #if self.quantization_param == 10:
        #    signs = torch.sign(v)
        #    v = torch.abs(v)
        #    v_base = (v - 1).clamp(min=0) * 10 + 5
        #    v_extra = (v >= 2).int() * 5
        #    return signs * (v_base + v_extra)
        #else:
        #    return v * self.quantization_param

        #return torch.sign(v) * torch.pow(torch.abs(v), self.quantization_param)

        global DEBUG_USE_MPEG
        ''' for mpeg compression '''
        if DEBUG_USE_MPEG:
            return torch.round(v)

        ''' normal solution '''
        return v * self.quantization_param


    def debug_get_mv_residual(self, input_image, refer_frame):
        estmv = self.opticFlow(input_image, refer_frame)
        mvfeature = self.mvEncoder(estmv)
        quant_mv = self.round_values(mvfeature)
        quant_mv_upsample = self.mvDecoder(self.unround_values(quant_mv))
        prediction, warpframe = self.motioncompensation(refer_frame, quant_mv_upsample)
        input_residual = input_image - prediction

        return estmv, input_residual

    def encode(self, input_image, refer_frame, return_z = False):
        """
        Parameters: 
            input_image: image tensor, with shape: (N, C, H, W)
            refer_frame: image tensor, with shape: (N, C, H, W)
        Returns:
            motion_vec: the encoded motion vector (after Quantization)
            residual: the encoded residual  (after Quantization)
        """
        estmv = self.opticFlow(input_image, refer_frame)
        mvfeature = self.mvEncoder(estmv)
        quant_mv = torch.round(mvfeature)
        #quant_mv = self.round_values(mvfeature)

        quant_mv_upsample = self.mvDecoder(quant_mv)
        #quant_mv_upsample = self.mvDecoder(self.unround_values(quant_mv))
        prediction, warpframe = self.motioncompensation(refer_frame, quant_mv_upsample)
        input_residual = input_image - prediction
        feature = self.resEncoder(input_residual)
        feature_renorm = feature
        #compressed_feature_renorm = torch.round(feature_renorm)
        compressed_feature_renorm = self.round_values(feature_renorm)
        
        if not return_z:
            return quant_mv, compressed_feature_renorm
        else:
            z = self.respriorEncoder(compressed_feature_renorm)
            #z = self.respriorEncoder(feature_renorm)
            compressed_z = torch.round(z)
            return quant_mv, compressed_feature_renorm, compressed_z

    def decode(self, refer_frame, motion_vec, residual):
        """
        parameter:
            refer_frame: the reference frame
            motion_vec: the encoded motion vector (after quantization)
            residual: the encoded residual (after quantization)
        returns:
            recon_image: the reconstructed image with shape N, C, H, W
        """
        #motion_vec = self.unround_values(motion_vec)
        residual = self.unround_values(residual)

        quant_mv_upsample = self.mvDecoder(motion_vec)
        prediction, warpframe = self.motioncompensation(refer_frame, quant_mv_upsample)
        recon_residual = self.resDecoder(residual)
        recon_image = prediction + recon_residual

        return recon_image
        
