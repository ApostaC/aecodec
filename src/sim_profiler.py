import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

from qmap.qmap_model import QmapModel
import pandas as pd
import pandas as pd
from utils.entropy_coder import GzipCoder, MPEGCoderWrapper
#from cabac_coder.cabac_coder import CABACCoder, CABACCoderTorchWrapper
import cv2
import os, sys
import subprocess as sp
import io
import shlex
from copy import deepcopy
from tqdm import tqdm
from dvc.dvc_model import DVCModel, DVCEntropyCoder
from torchvision.transforms.functional import to_tensor
import torch
import numpy as np
import utils.entropy_coder as EC
import utils.packet as P
from torchvision.utils import save_image
from PIL import Image, ImageFile, ImageFilter
from pipeline import FinalEncoder, FinalDecoder, DebugCoderPFrame
from skimage.metrics import peak_signal_noise_ratio
from utils.video_loader import VideoDataset
import time
from skimage.metrics import structural_similarity as ssim
from quant import getDefaultQuantizer, FQuantizer
from scipy.stats import pearsonr
from queue import PriorityQueue
from dataclasses import dataclass
import PIL


def print_usage():
    print(
        f"Usage: {sys.argv[0]} <video_file> <output file> <mode> [dvc_model]"
        f""
        f"  mode = mpeg | ae"
    )
    exit(1)

def PSNR(Y1_raw, Y1_com):
    Y1_com = Y1_com.to(Y1_raw.device)
    log10 = torch.log(torch.FloatTensor([10])).squeeze(0).to(Y1_raw.device)
    train_mse = torch.mean(torch.pow(Y1_raw - Y1_com, 2))
    quality = 10.0*torch.log(1/train_mse)/log10
    return quality

def SSIM(Y1_raw, Y1_com):
    y1 = Y1_raw.permute([1,2,0]).cpu().detach().numpy()
    y2 = Y1_com.permute([1,2,0]).cpu().detach().numpy()
    return ssim(y1, y2, multichannel=True)

METRIC_FUNC = PSNR

def read_video_into_frames(video_path, frame_size=None, nframes=1000):
    """
    Input:
        video_path: the path to the video
        frame_size: resize the frame to a (width, height), if None, it will not do resize
        nframes: number of frames
    Output:
        frames: a list of PIL images
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        if np.sum(img) == 0:
            continue

        img = Image.fromarray(img)
        if frame_size is not None:
            img = img.resize(frame_size)
        else:
            ''' pad to nearest 64 '''
            padsz = 64
            w, h = img.size
            pad_w = int(np.ceil(w / padsz) * padsz)
            pad_h = int(np.ceil(h / padsz) * padsz)
            img = img.resize((pad_w, pad_h))
        frames.append(img)

        if len(frames) >= nframes:
            break
    print("Resizing image to", frames[-1].size)
    return frames


class EncodedFrame:
    """
    self.code is torch.tensor
    """
    def __init__(self, code, shapex, shapey, frame_type, frame_id):
        self.code = code
        self.shapex = shapex
        self.shapey = shapey
        self.frame_type = frame_type
        self.frame_id = frame_id
        self.loss_applied = False

    def apply_loss(self, loss_ratio, blocksize = 100):
        """
        default block size is 100
        """
        leng = torch.numel(self.code)
        nblocks = (leng - 1) // blocksize + 1

        rnd = torch.rand(nblocks).to(self.code.device)
        rnd = (rnd > loss_ratio).long()
        rnd = rnd.repeat_interleave(blocksize)
        rnd = rnd[:leng].reshape(self.code.shape)
        self.code = self.code * rnd

    def apply_loss_determ(self, loss_prob):
        REPEATS=64
        nelem = torch.numel(self.code)
        group_len = int((nelem - 1) // REPEATS + 1)
        rnd = torch.rand(group_len).cuda()
        rnd = (rnd > loss_prob).long()
        rnd = rnd.repeat(REPEATS)[:nelem]
        rnd = rnd.reshape(self.code.shape)
        self.code = self.code * rnd

    def apply_mask(self, mask):
        self.code = self.code * mask

    def np_code(self):
        """
        return the code in flattened numpy array
        """
        return self.code.cpu().detach().numpy().flatten()

@dataclass(order=True)
class CodeSeg:
    def __init__(self, code, start, end):
        self.start = start
        self.end = end
        self.is_orig = True
        self.orig_seg = None
        self.bpp_grad_local = None
        self.psnr_grad_delta_local = None
        self.psnr_grad_local = None
        self.mean = torch.mean(code[start:end]).item()
        self.copies = []
        self.is_dropped = False
        self.is_recovered = True

    def __eq__(self, other):
        if isinstance(other, CodeSeg):
            return (self.start == other.start and self.end == other.end)
        return False

    def get_code(self, code):
        if self.is_orig:
            if self.is_recovered:
                return code[self.start:self.end]
            else:
                # import pdb; pdb.set_trace()
                return torch.full((self.end - self.start,), self.mean)
        else:
            return self.orig_seg.get_code(code)

    def recover(self, config):
        if self.is_orig:
            if self.is_dropped:
                if len(self.copies) > 0:
                    if config.debug:
                        print("--------> TRYING TO RECOVER")
                        self.print_full()
                for seg in self.copies:
                    if not seg.is_dropped:
                        self.is_recovered = True
                        if config.debug: print("--------> SUCCESSFULLY RECOVER")
                        break
        else:
            self.is_orig = True

    def print_full(self):
        copies_str = "["
        for seg in self.copies:
            copies_str += str(seg.start)+'->'+str(seg.end)+'|'+str(seg.is_dropped)+', '
        copies_str += ']'
        print(' | Range: '+str(self.start)+'->'+str(self.end)+' # m='+str(self.mean)+'\n'+
        ' | Orig: '+str(self.is_orig)+', '+copies_str+'\n'+
        ' | Grads: '+str(self.bpp_grad_local)+
        ', '+str(self.psnr_grad_delta_local)+', '+str(self.psnr_grad_local)+'\n'+
        ' | D/R: '+str(self.is_dropped)+', '+str(self.is_recovered))

    def print_short(self):
        print(' | Range: '+str(self.start)+'->'+str(self.end)+' # m='+str(self.mean))

class AEModel:
    def __init__(self, qmap_coder: QmapModel, dvc_coder: DVCModel, only_P=True):
        self.qmap_coder = qmap_coder
        self.dvc_coder = dvc_coder
        #self.entropy_coder = GzipCoder(dtype=np.int8)
        self.entropy_coder = DVCEntropyCoder(dvc_coder)
        #self.entropy_coder = CABACCoderTorchWrapper(0.1)
        #self.entropy_coder = MPEGCoderWrapper()
        #self.entropy_coder.set_qp(0)
        #self.entropy_coder.impl.enable_debug()

        self.reference_frame = None
        self.frame_counter = 0
        self.gop = 8
        self.only_P = only_P

        self.debug_output_dir = None

        if not only_P:
            raise RuntimeError("Now only support only_P = True for AEModels! Becase torchac only works on DVC")

        #''' debug '''
        #self.entropy_code_mv = DVCEntropyCoder(dvc_coder)
        #self.entropy_coder_res = MPEGCoderWrapper()
        #self.entropy_coder_res.set_qp(0)
        #self.entropy_coder_res.impl.enable_debug()

    def set_gop(self, gop):
        self.gop = gop

    def set_quantization_param(self, q):
        self.dvc_coder.set_quantization_param(q)
        #self.entropy_coder.impl.set_qp(q)
        #self.entropy_coder_res.set_qp(q)

    def get_quantization_param(self):
        return self.dvc_coder.get_quantization_param()

    def set_debug_output_dir(self, path):
        self.debug_output_dir = path

    def encode_frame(self, frame, isIframe = False):
        """
        Input:
            frame: the PIL image
        Output:
            eframe: encoded frame, code is torch.tensor on GPU
        Note:
            this function will NOT update the reference
        """
        self.frame_counter += 1
        frame = to_tensor(frame)
        if isIframe:
            code, shapex, shapey = self.qmap_coder.encode(frame)
            eframe = EncodedFrame(code, shapex, shapey, "I", self.frame_counter)
            return eframe
        else:
            assert self.reference_frame is not None
            code, shapex, shapey, z = self.dvc_coder.encode(frame, self.reference_frame, return_z = True)
            eframe = EncodedFrame(code, shapex, shapey, "P", self.frame_counter)
            return eframe, z

    def decode_frame(self, eframe:EncodedFrame):
        """
        Input:
            eframe: the encoded frame (EncodedFrame object)
        Output:
            frame: the decoded frame in torch.tensor (3,h,w) on GPU, which can be used as ref frame
        Note:
            this function will NOT update the reference
        """
        if eframe.frame_type == "I":
            out = self.qmap_coder.decode(eframe.code, eframe.shapex, eframe.shapey)
            return out
        else:
            assert self.reference_frame is not None
            out = self.dvc_coder.decode(eframe.code, self.reference_frame, eframe.shapex, eframe.shapey)
            return out

    def encode_with_loss(self, frames, losses, need_bpp = False):
        """
        Input:
            frames: PIL images
            loss: packet loss ratio for the each frame
        Output:
            list of METRIC_FUNC and list of BPP
        """
        bpps = []
        psnrs = []
        for idx, frame in enumerate(tqdm(frames)):
        #for idx, frame in enumerate(frames):
            loss = losses[idx]
            ''' Encode the frame '''
            if idx % self.gop == 0:
                ''' I FRAME '''
                #eframe = self.encode_frame(frame, True)
                self.update_reference(to_tensor(frame))
                continue
            else:
                #self.update_reference(to_tensor(frames[idx-1]))
                eframe, z = self.encode_frame(frame)


            # compute bpp
            if need_bpp and idx % self.gop != 0: # not I frame
                bs, tot_size = self.entropy_coder.entropy_encode(eframe.code, eframe.shapex, eframe.shapey, z)
                w, h = frame.size
                bpp = tot_size * 8 / (w * h)
                bpps.append(bpp)

            # apply loss
            #eframe.apply_loss_determ(loss)
            eframe.apply_loss(loss, 1)

            # decode frame
            decoded = self.decode_frame(eframe)
            self.update_reference(decoded)

            # compute psnr
            tframe = to_tensor(frame)
            psnr = METRIC_FUNC(tframe, decoded)
            psnrs.append(psnr)

            if self.debug_output_dir is not None:
                save_image(decoded, f"{self.debug_output_dir}/{idx}-recon.png")
                save_image(tframe, f"{self.debug_output_dir}/{idx}-origin.png")

        return psnrs, bpps

    def create_mask(self, code, loss_ratio, blocksize = 100):
        """
        default block size is 100
        """
        leng = torch.numel(code)
        nblocks = (leng - 1) // blocksize + 1

        rnd = torch.rand(nblocks).to(code.device)
        rnd = (rnd > loss_ratio).long()
        rnd = rnd.repeat_interleave(blocksize)
        rnd = rnd[:leng].reshape(code.shape)
        return rnd

    def get_psnr_saliency(self, code, reference_frame, shapex, shapey, orig_image):
        mv_grad, res_grad = self.dvc_coder.get_saliency(code, reference_frame,
                                                        shapex, shapey, orig_image)
        mv_grad = torch.flatten(mv_grad)
        res_grad = torch.flatten(res_grad)
        psnr_grad = torch.cat([mv_grad, res_grad])
        return psnr_grad

    def get_bpp_saliency(self, code, shapex, shapey, z):
        mv_bpp_grad, res_bpp_grad = self.entropy_coder.entropy_encode_saliency(
                                code, shapex, shapey, z)
        mv_bpp_grad = torch.flatten(mv_bpp_grad)
        res_bpp_grad = torch.flatten(res_bpp_grad)
        bpp_grad = torch.cat([mv_bpp_grad, res_bpp_grad])
        return bpp_grad

    def reverse_transform(self, segs, config):
        for seg in segs:
            seg.recover(config)

    def split(self, code, refer_frame, shapex, shapey, orig_image, z, config):
        seg_size = config.seg_size
        leng = torch.numel(code)
        # import pdb; pdb.set_trace()
        psnr_grad = self.get_psnr_saliency(code, refer_frame, shapex, shapey, orig_image)
        # bpp_grad = self.get_bpp_saliency(code, shapex, shapey, z)
        mvsize = np.prod(shapex)
        ressize = np.prod(shapey)
        assert mvsize + ressize == leng
        segs_mv = []
        segs_res = []
        for l in range(0, mvsize, seg_size):
            left = l
            right = np.min([l+seg_size, mvsize])
            seg = CodeSeg(code, left, right)
            grad_mask = torch.zeros(leng).to(code.device)
            grad_mask[left:right] = 1
            delta = code * grad_mask
            delta[left:right] = torch.mean(code[left:right])
            # bpp_grad_local = torch.sum((bpp_grad * grad_mask).pow(2))
            psnr_grad_delta_local = torch.sum((psnr_grad * delta * grad_mask).pow(2))
            psnr_grad_local = torch.sum((psnr_grad * grad_mask).pow(2))
            # seg.bpp_grad_local = bpp_grad_local.item()
            seg.psnr_grad_delta_local = psnr_grad_delta_local.item()
            seg.psnr_grad_local = psnr_grad_local.item()
            segs_mv.append(seg)
        for l in range(mvsize, mvsize + ressize, seg_size):
            left = l
            right = np.min([l+seg_size, leng])
            seg = CodeSeg(code, left, right)
            grad_mask = torch.zeros(leng).to(code.device)
            grad_mask[left:right] = 1
            delta = code * grad_mask
            delta[left:right] = torch.mean(code[left:right])
            # bpp_grad_local = torch.sum((bpp_grad * grad_mask).pow(2))
            psnr_grad_delta_local = torch.sum((psnr_grad * delta * grad_mask).pow(2))
            psnr_grad_local = torch.sum((psnr_grad * grad_mask).pow(2))
            # seg.bpp_grad_local = bpp_grad_local.item()
            seg.psnr_grad_delta_local = psnr_grad_delta_local.item()
            seg.psnr_grad_local = psnr_grad_local.item()
            segs_res.append(seg)
        return segs_mv, segs_res

    def getNextCleanSegs(self, queue, count, excludedSegs, requires_no_copies):
        segs = []
        for i in range(count):
            found = False
            while True:
                seg = queue.get()[1]
                if seg.is_orig and (not seg in excludedSegs):
                    if (not requires_no_copies) or len(seg.copies) == 0:
                        found = True; break
            if found: segs.append(seg)
        return segs

    def dup_seg (self, seg_src, seg_dst):
        seg_dst.is_orig = False
        seg_dst.orig_seg = seg_src
        seg_src.copies.append(seg_dst)

    def transform(self, segs, config, stats):
        topk = config.num_top_segs
        num_copies = config.num_copies
        # import pdb; pdb.set_trace()
        # by_bpp_grad_local = PriorityQueue()
        by_psnr_grad_delta_local = PriorityQueue()
        by_psnr_grad_local = PriorityQueue()
        by_mean_local = PriorityQueue()
        for i in range(len(segs)-1): # the last seg doesn't participate shuffling
            seg = segs[i]
            # import pdb; pdb.set_trace()
            # by_bpp_grad_local.put((seg.bpp_grad_local, seg))
            by_psnr_grad_delta_local.put((seg.psnr_grad_delta_local, seg))
            by_psnr_grad_local.put((-seg.psnr_grad_local, seg))
            by_mean_local.put((-np.abs(seg.mean), seg))
        # ideal block to be replaced: low psnr_grad_delta_local
        # ideal block to replace with: high psnr_grad_local, low bpp_grad_local
        if config.debug:
            by_psnr_grad_local_copy = PriorityQueue()
            for i in range(len(segs)-1):
                seg = segs[i]
                by_psnr_grad_local_copy.put((-seg.psnr_grad_local, seg))
            for i in range(topk):
                seg = by_psnr_grad_local_copy.get()[1]
                stats.record_top_seg(str(seg.start))
        # use static for testing
        static_top_segs = [60448, 60416, 46048, 60384, 60320, 45888, 60224, 45920]
        use_static = False
        if int(np.max(static_top_segs)/config.seg_size) > len(segs):
            use_static = False

        if use_static:
            srcSegs = []
            for i in range(topk):
                srcSegs.append(segs[int(static_top_segs[i]/config.seg_size)])
            for i in range(topk):
                srcSeg = srcSegs[i]
                # dstSegs = self.getNextCleanSegs(by_psnr_grad_delta_local, num_copies, srcSegs)
                # dstSegs = self.getNextCleanSegs(by_bpp_grad_local, num_copies, srcSegs)
                dstSegs = self.getNextCleanSegs(by_mean_local, num_copies, srcSegs, True)
                for dstSeg in dstSegs:
                    # print("==============")
                    # srcSeg.print_short()
                    # print("COPIED TO")
                    # dstSeg.print_short()
                    # print("==============")
                    self.dup_seg(srcSeg, dstSeg)
        else:
            for i in range(topk):
                srcSegs = self.getNextCleanSegs(by_psnr_grad_local, 1, [], False)
                srcSeg = srcSegs[0]
                # dstSegs = self.getNextCleanSegs(by_psnr_grad_delta_local, num_copies, srcSegs)
                # dstSegs = self.getNextCleanSegs(by_bpp_grad_local, num_copies, srcSegs)
                dstSegs = self.getNextCleanSegs(by_mean_local, num_copies, srcSegs, True)
                for dstSeg in dstSegs:
                    # print("==============")
                    # srcSeg.print_short()
                    # print("COPIED TO")
                    # dstSeg.print_short()
                    # print("==============")
                    self.dup_seg(srcSeg, dstSeg)

    def apply_loss_segs(self, segs, loss, config):
        count_dropped = 0
        np.random.seed(10)
        for i in range(len(segs)):
            seg = segs[i]
            # if (i+7) % (int(1.0/(loss+0.0000001))) == 0:
            if np.random.uniform() < loss:
                seg.is_dropped = True
                seg.is_recovered = False
                count_dropped += 1
        if config.debug: print('Loss stats: '+str(count_dropped)+'/'+str(len(segs)))

    def segs_to_code(self, segs, code):
        leng = torch.numel(code)
        new_code = torch.zeros(leng)
        for seg in segs:
            new_code[seg.start:seg.end] = seg.get_code(code)
        return new_code

    def postprocessing_v2(self, eframe, code_copy, frame, z, loss, config, stats):
        segs_mv, segs_res = self.split(eframe.code, self.reference_frame,
                                    eframe.shapex, eframe.shapey,
                                    to_tensor(frame).to(self.dvc_coder.device), z, config)
        # self.transform(segs, num_copies = copies_count)
        self.transform(segs_mv, config, stats)
        self.transform(segs_res, config, stats)
        segs = np.concatenate([segs_mv, segs_res])
        eframe.code = self.segs_to_code(segs, code_copy)
        zz = self.dvc_coder.encode_z(eframe.code[np.prod(eframe.shapex):].to(self.dvc_coder.device), eframe.shapey)
        zz = torch.round(zz)
        bs, tot_size = self.entropy_coder.entropy_encode(eframe.code,
                                eframe.shapex, eframe.shapey, zz)
        w, h = frame.size
        bpp = tot_size * 8 / (w * h)

        # app seg-level losses
        self.apply_loss_segs(segs, loss, config)
        # reverse transform
        self.reverse_transform(segs, config)
        eframe.code = self.segs_to_code(segs, code_copy)
        # decode frame
        decoded = self.decode_frame(eframe)

        # compute psnr
        tframe = to_tensor(frame)
        psnr = METRIC_FUNC(tframe, decoded)
        return psnr.item(), bpp, decoded

    def encode_with_loss_optimized_v2(self, frames, losses, config, need_bpp = False):
        """
        Input:
            frames: PIL images
            loss: packet loss ratio for the each frame
        Output:
            list of METRIC_FUNC and list of BPP and a OptimizeStats
        """
        bpps = []
        psnrs = []
        stats = OptimizeStats()
        for idx, frame in enumerate(tqdm(frames)):

            loss = losses[idx]
            check_ids = [500, 600, 700] # debug
            ''' Encode the frame '''
            if idx % self.gop == 0:
                ''' I FRAME '''
                #eframe = self.encode_frame(frame, True)
                self.update_reference(to_tensor(frame))
                continue
            else:
                #self.update_reference(to_tensor(frames[idx-1]))
                # testing the effect of RGB order
                # R, G, B = frame.split()
                # frame = PIL.Image.merge("RGB", (B, G, R))
                eframe, z = self.encode_frame(frame)
            leng = torch.numel(eframe.code)
            if config.debug: print("new frame, # of vals = ", leng)
            code_copy = eframe.code.clone()

            psnr, bpp, decoded = self.postprocessing_v2(eframe, code_copy, frame,
                                z, loss, config, stats)
            if config.debug: print("stats: ", bpp, psnr)

            self.update_reference(decoded)
            psnrs.append(psnr)
            bpps.append(bpp)

            if self.debug_output_dir is not None:
                save_image(decoded, f"{self.debug_output_dir}/{idx}-recon.png")
                save_image(tframe, f"{self.debug_output_dir}/{idx}-origin.png")

        return psnrs, bpps, stats

    def grouping(segs, num_groups = 20):
        groups = []
        segs_per_group = int(len(segs)/num_groups) \
                        if (len(segs) % num_groups == 0) \
                        else int(len(segs)/num_groups)+1
        by_psnr_grad_local = PriorityQueue()
        for seg in segs:
            by_psnr_grad_local.put((-seg.psnr_grad_local, seg))
        for i in range(num_groups):
            current_group = []
            current_num_seg = len(segs) - i * segs_per_group
            for j in range(current_num_seg):
                seg = by_psnr_grad_local.get()[1]
                current_group.append(seg)
            groups.append(current_group)
        return groups


    def encode_frame_v3(self, eframe, code_copy, frame, z, loss, copies_count):
        segs_mv, segs_res = self.split(eframe.code, self.reference_frame,
                                    eframe.shapex, eframe.shapey,
                                    to_tensor(frame).to(self.dvc_coder.device), z)
        groups_mv = grouping(segs_mv)
        groups_res = grouping(segs_res)
        assert len(groups_mv) == len(groups_res)
        num_groups = len(groups_mv)

        # top k groups are fec proctected with 30% redundancy
        num_fec_groups = 3
        fec_rate = 0.3
        bpp = 0
        for i in range(num_groups):
            code_mv_per_group = self.segs_to_code(groups_mv[i], code_copy)
            code_res_per_group = self.segs_to_code(groups_res[i], code_copy)

            bs, tot_size_per_group = self.entropy_coder.entropy_encode(
                                    torch.cat([code_mv_per_group, code_res_per_group]),
                                    torch.numel(code_mv_per_group),
                                    torch.numel(code_res_per_group), z)
            w, h = frame.size
            bpp_per_group = tot_size_per_group * 8 / (w * h)
            if i < num_fec_groups:
                bpp_per_group *= 1 + fec_rate
            bpp += bpp_per_group
        self.apply_loss_segs(segs, loss)
        for i in range(num_fec_groups):
            dropped_seg_count = 0
            for seg in groups_mv[i]:
                if seg.is_dropped: dropped_seg_count += 1
            for seg in groups_res[i]:
                if seg.is_dropped: dropped_seg_count += 1
            # if float(dropped_seg_count/(len(groups_mv)+len(groups_res))) >
        return 0, 0

    # FEC optimized
    def encode_with_loss_optimized_v3(self, frames, losses, need_bpp = False):
        """
        Input:
            frames: PIL images
            loss: packet loss ratio for the each frame
        Output:
            list of METRIC_FUNC and list of BPP
        """
        bpps = []
        psnrs = []
        for idx, frame in enumerate(tqdm(frames)):
            loss = losses[idx]
            check_ids = [500, 600, 700] # debug
            ''' Encode the frame '''
            if idx % self.gop == 0:
                ''' I FRAME '''
                #eframe = self.encode_frame(frame, True)
                self.update_reference(to_tensor(frame))
                continue
            else:
                #self.update_reference(to_tensor(frames[idx-1]))
                eframe, z = self.encode_frame(frame)
            leng = torch.numel(eframe.code)
            print("new frame, # of vals = ", leng)
            code_copy = eframe.code.clone()

            psnr, bpp = encode_frame_v3(eframe, code_copy, frame, z, loss)

            self.update_reference(decoded)
            psnrs.append(psnr)
            bpps.append(bpp)

            if self.debug_output_dir is not None:
                save_image(decoded, f"{self.debug_output_dir}/{idx}-recon.png")
                save_image(tframe, f"{self.debug_output_dir}/{idx}-origin.png")

        return psnrs, bpps

    def encode_with_loss_optimized_with_gradient(self, frames, losses, need_bpp = False):
        """
        Input:
            frames: PIL images
            loss: packet loss ratio for the each frame
        Output:
            list of METRIC_FUNC and list of BPP
        """
        bpps = []
        psnrs = []
        for idx, frame in enumerate(tqdm(frames)):
        #for idx, frame in enumerate(frames):
            loss = losses[idx]
            ''' Encode the frame '''
            if idx % self.gop == 0:
                ''' I FRAME '''
                #eframe = self.encode_frame(frame, True)
                self.update_reference(to_tensor(frame))
                continue
            else:
                #self.update_reference(to_tensor(frames[idx-1]))
                eframe, z = self.encode_frame(frame)

            # apply loss
            #eframe.apply_loss_determ(loss)
            # eframe.apply_loss(loss, 1)

            blocksize = 128
            leng = torch.numel(eframe.code)
            print("new frame, # of vals = ", leng)

            ''' calculate gradients'''
            psnr_grad = self.get_psnr_saliency(eframe.code, self.reference_frame,
                                          eframe.shapex, eframe.shapey,
                                          to_tensor(frame).to(self.dvc_coder.device))

            ''' calculate bpp and gradients'''
            # import pdb; pdb.set_trace()
            if need_bpp and idx % self.gop != 0: # not I frame
                bs, tot_size = self.entropy_coder.entropy_encode(eframe.code,
                                        eframe.shapex, eframe.shapey, z)
                bpp_grad = self.get_bpp_saliency(eframe.code, eframe.shapex, eframe.shapey, z)
                # import pdb; pdb.set_trace()
                # with open("/tmp/scatter-{}.csv".format(idx), 'w') as f:
                #     for i in range(0, leng, 1):
                #         f.write(str(np.abs(psnr_grad[i].item()))+\
                #         ','+str(np.abs(bpp_grad[i].item()))+\
                #         ','+str(np.abs(eframe.code[i].item()))+'\n')
                f = open("/tmp/scatter-{}-blocks.csv".format(idx), 'w')

                for r in range(0, leng, blocksize):
                    left = r
                    right = np.min([r+blocksize, leng])
                    grad_mask = torch.zeros(leng).to(eframe.code.device)
                    grad_mask[left:right] = 1
                    bpp_grad_local = torch.sum((bpp_grad * grad_mask).pow(2))
                    delta = eframe.code * grad_mask
                    delta[left:right] = torch.mean(eframe.code[left:right])
                    psnr_grad_delta_local = torch.sum((psnr_grad * delta * grad_mask).pow(2))
                    psnr_grad_local = torch.sum((psnr_grad * grad_mask).pow(2))
                    # ideal block to be replaced: low psnr_grad_delta_local
                    # ideal block to replace with: high psnr_grad_local, low bpp_grad_local
                    f.write(str(np.abs(psnr_grad_delta_local.item()))+\
                        ','+str(np.abs(psnr_grad_local.item()))+\
                        ','+str(np.abs(bpp_grad_local.item()))+'\n')
                f.close()

                w, h = frame.size
                bpp = tot_size * 8 / (w * h)
                bpps.append(bpp)

            code_copy = eframe.code.clone()
            grad_total_arr = []
            psnr_arr = []
            for i in range(5):
                mask = self.create_mask(eframe.code, loss, blocksize)
                ''' analyzing the impact of apply_mask '''
                eframe.code = code_copy
                grad_mask = torch.ones(leng).to(eframe.code.device)-mask
                loss_grad = psnr_grad * grad_mask
                loss_grad_sum_without_code = torch.sum(loss_grad.pow(2))*np.power(10,13)
                loss_grad = psnr_grad * grad_mask * eframe.code
                loss_grad_sum_with_code = torch.sum(loss_grad.pow(2))*np.power(10,13)
                # loss_grad_sum_with_code = torch.sum(loss_grad)*np.power(10,6)

                # print("before:", torch.sum(eframe.code))
                eframe.apply_mask(mask)
                # print("after:", torch.sum(eframe.code))

                # decode frame
                decoded = self.decode_frame(eframe)

                # compute psnr
                tframe = to_tensor(frame)
                psnr = METRIC_FUNC(tframe, decoded)
                print(torch.sum(grad_mask).item(), \
                            loss_grad_sum_without_code.item(), \
                            loss_grad_sum_with_code.item(), psnr.item())
                psnr_arr.append(psnr.item())
                grad_total_arr.append(loss_grad_sum_with_code.item())
            # print("correlation = ", np.correlate(grad_total_arr, psnr_arr))
            print("correlation = ", pearsonr(grad_total_arr, psnr_arr))
            '''
            print cdf of grads
            '''
            # import pdb; pdb.set_trace()
            # perm = torch.randperm(grad.size(0))
            # idx = perm[:1000]
            # samples = grad[idx]
            # samples = sorted(samples)
            # for r in range(0, 40, 1):
            #     print(samples[int(len(samples)*r/40)].item()*np.power(10,10), r)

            # grad_total_arr = []
            # for r in range(0, leng, blocksize):
            #     grad_mask = torch.zeros(leng).to(eframe.code.device)
            #     grad_mask[r:np.min([r+blocksize, leng])] = 1
            #     loss_grad = psnr_grad * grad_mask * eframe.code
            #     grad_total_arr.append(torch.sum(loss_grad.pow(2))*np.power(10,13))
            # grad_total_arr = sorted(grad_total_arr)
            # print("# of blocks = ", len(grad_total_arr))
            # for r in range(0, 40, 1):
            #     print(grad_total_arr[int(len(grad_total_arr)*r/40)].item(), r)


            self.update_reference(decoded)
            psnrs.append(psnr)

            if self.debug_output_dir is not None:
                save_image(decoded, f"{self.debug_output_dir}/{idx}-recon.png")
                save_image(tframe, f"{self.debug_output_dir}/{idx}-origin.png")

        return psnrs, bpps

    def debug_encode_video(self, frames, use_mpeg=True):
        """
        Input:
            frames: PIL images
        Output:
            list of METRIC_FUNC and list of BPP
        """
        bpps = []
        psnrs = []
        test_iter = tqdm(frames)
        for idx, frame in enumerate(test_iter):
            if idx % self.gop == 0:
                self.update_reference(to_tensor(frame))
                continue
            else:
                eframe, z = self.encode_frame(frame)

                ''' check the grad '''
                grad_mv, grad_res = self.dvc_coder.get_saliency(eframe.code, self.reference_frame, eframe.shapex, eframe.shapey, to_tensor(frame).to(self.dvc_coder.device))

                ''' debug: normalize grad_res to 0, 255 and save it to image '''

            mvsize = np.prod(eframe.shapex)
            ressize = np.prod(eframe.shapey)

            mv = eframe.code[:mvsize].reshape(eframe.shapex)
            res = eframe.code[mvsize:]
            print("Residual range: ", torch.max(res), torch.min(res))

            ''' compress mv and get size '''
            if self.entropy_coder_res.impl.cmd is None:
                self.entropy_coder_res.fit_code_size(ressize)
            _, mv_size_cmp = self.entropy_code_mv.compress_mv(mv)


            ''' debug '''
            debug_quantizer = getDefaultQuantizer(150)
            self.entropy_code_mv.set_quantizer(debug_quantizer)
            _, mv_size_cmp = self.entropy_code_mv.compress_mv(mv, True)

            if use_mpeg:
                ''' use mpeg to compress residual '''
                #res_sign = torch.sign(res)
                #res = torch.abs(res)
                bs, res_size_cmp, vmin, vmax = self.entropy_coder_res.debug_entropy_encode(res, eframe.shapex, eframe.shapey, z)
            else:
                ''' use torchac to compress residual '''
                sigma = self.dvc_coder.model.respriorDecoder(z)
                _, res_size_cmp = self.entropy_code_mv.compress_res(res.reshape(eframe.shapey), sigma)

            tot_size = mv_size_cmp + res_size_cmp
            #tot_size = res_size_cmp
            print("Sizes {} = {} + {}".format(mv_size_cmp + res_size_cmp, mv_size_cmp, res_size_cmp))

            ''' decode '''
            eframe2 = deepcopy(eframe)
            ''' debug '''
            quant_mv = debug_quantizer.quantize(eframe.code[:mvsize])
            eframe2.code[:mvsize] = debug_quantizer.dequantize(quant_mv)

            if use_mpeg:
                dec = self.entropy_coder_res.debug_entropy_decode(bs, vmin, vmax)
                #dec = dec * res_sign.to(dec.device)
                eframe2.code[mvsize:] = dec
            print("Changed elements:", torch.sum((eframe2.code.to(eframe.code.device) != eframe.code).to(torch.int8)))

            decoded = self.decode_frame(eframe2)
            self.update_reference(decoded)

            ''' compute psnr '''
            tframe = to_tensor(frame)
            psnr = float(METRIC_FUNC(tframe, decoded))
            psnrs.append(psnr)

            ''' compute bpp '''
            w, h = frame.size
            bpp = tot_size * 8 / (w * h)
            bpps.append(bpp)

            ''' debug '''
            if use_mpeg:
                tmp = self.entropy_coder_res.impl.get_debug_info()

            #import debug
            #debug.save_np_to_csv(tmp[0], "../debug/tmp0.csv")
            #debug.save_np_to_csv(tmp[1], "../debug/tmp1.csv")
            #debug.save_np_to_csv(tmp[2], "../debug/tmp2.csv")
            #debug.save_np_to_csv(tmp[3], "../debug/tmp3.csv")


            test_iter.set_description(f"bpp:{np.mean(bpps):.4f}, psnr:{np.mean(psnrs):.4f}")
            #import pdb
            #pdb.set_trace()

        return psnrs, bpps

    def encode_video(self, frames, use_mpeg=True):
        """
        Input:
            frames: PIL images
        Output:
            list of METRIC_FUNC and list of BPP
        """
        import dvc.net
        dvc.net.DEBUG_USE_MPEG = True
        bpps = []
        psnrs = []
        test_iter = tqdm(frames)
        for idx, frame in enumerate(test_iter):
            # encode the frame
            if idx % self.gop == 0:
                ''' I FRAME '''
                self.update_reference(to_tensor(frame))
                continue
            else:
                eframe, z = self.encode_frame(frame)

            # decode frame
            w, h = frame.size
            decoded = self.decode_frame(eframe)
            self.update_reference(decoded)

            # compute psnr
            if idx % self.gop != 0:
                tframe = to_tensor(frame)
                psnr = float(METRIC_FUNC(tframe, decoded))
                psnrs.append(psnr)


            # compute bpp
            #if idx % self.gop != 0 and np.random.uniform() < 0.3:
            # import pdb; pdb.set_trace()
            if idx % self.gop != 0:
                ''' whole frame compression '''
                bs, tot_size = self.entropy_coder.entropy_encode(eframe.code, \
                                        eframe.shapex, eframe.shapey, z)
                w, h = frame.size
                bpp = tot_size * 8 / (w * h)
                bpps.append(bpp)

            test_iter.set_description(f"bpp:{np.mean(bpps):.4f}, psnr:{np.mean(psnrs):.4f}")

        return psnrs, bpps

    def update_reference(self, ref_frame):
        """
        Input:
            ref_frame: reference frame in torch.tensor with size (3,h,w). On GPU
        """
        self.reference_frame = ref_frame

    def get_avg_freeze_psnr(self, frames):
        res = []
        for idx, frame in enumerate(frames[2:]):
            img1 = to_tensor(frame)
            img2 = to_tensor(frames[idx-2])
            res.append(METRIC_FUNC(img1, img2))
        return float(np.mean(res))

class MPEGModel:
    def __init__(self, gop, only_P=True, use_265=True):
        self.qp = 15
        self.gop = gop
        self.only_P = only_P
        self.use_265 = use_265

    def set_qp(self, q):
        self.qp = q

    def get_qp(self):
        return self.qp

    def get_avg_freeze_psnr(self, frames):
        res = []
        for idx, frame in enumerate(frames[1:]):
            img1 = to_tensor(frame)
            img2 = to_tensor(frames[idx-1])
            res.append(METRIC_FUNC(img1, img2))
            print("res:", res[-1])
        return float(np.mean(res))


    def encode_video(self, frames):
        """
        Input:
            frames: list of PIL images
        Output:
            psnrs: psnr for each frame
            bpp: average BPP of the encoded video
        """
        assert self.qp is not None
        imgByteArr = io.BytesIO()
        fps = 25
        output_filename = f'/tmp/output-jj-{np.random.randint(0, 10000)}.mp4'
        width, height = frames[0].size
        cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p -preset veryfast -tune zerolatency -crf {self.qp} -g {self.gop} -sc_threshold 0 -loglevel debug {output_filename}'
        if self.use_265:
            cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx265 -pix_fmt yuv420p -preset veryfast -tune zerolatency -x265-params "crf={self.qp}:keyint={self.gop}:verbose=1" -sc_threshold 0 -loglevel debug {output_filename}'


        process = sp.Popen(shlex.split(cmd), stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT)
        for img in tqdm(frames):
            process.stdin.write(np.array(img).tobytes())
        process.stdin.close()
        process.wait()
        process.terminate()

        if self.only_P:
            bpp_res = sp.check_output(f'ffprobe -show_frames {output_filename} | grep "pkt_size\|pict_type" | grep "=P" -B 1 | grep "pkt_size"',
                                            stderr=open("/dev/null","w"), shell=True, encoding="utf-8")
        else:
            bpp_res = sp.check_output(f"ffprobe -show_frames {output_filename} | grep pkt_size", stderr=open("/dev/null","w"), shell=True, encoding="utf-8")
        temp = bpp_res.split("\n")
        temp.remove("")
        bpps = list(map(lambda s: int(s.split("=")[1]), temp))
        bpps = list(map(lambda B: B * 8 / (height * width), bpps))

        # get psnr
        psnrs = []

        clip = []
        cap = cv2.VideoCapture(output_filename)
        while(cap.isOpened()):
            ret, img = cap.read()
            if ret != True:break
            clip.append(to_tensor(img))

        for i, frame in enumerate(frames):
            Y1_raw = to_tensor(frame)
            Y1_com = clip[i]
            psnrs += [METRIC_FUNC(Y1_raw, Y1_com)]
        return psnrs, bpps



def init_ae_model(dvc_path=None) -> AEModel:
    ROOT_DIR="/datamirror/yihua98/projects/autoencoder_testbed"
    qmap_config_template = {
            "N": 192,
            "M": 192,
            "sft_ks": 3,
            "name": "default",
            "path": f"{ROOT_DIR}/models/qmap/2M_itrs.pt",
            "quality": 1,
        }
    if dvc_path is None:
        dvc_path = f"{ROOT_DIR}/models/dvc/yihua_test-3.model"
    dvc_config_template = { "path":  dvc_path}
    #dvc_config_template = { "path": f"{ROOT_DIR}/models/dvc/loss_30_big.model" }
    #dvc_config_template = { "path": f"{ROOT_DIR}/models/dvc/loss_30.model" }
    #dvc_config_template = { "path": f"{ROOT_DIR}/models/dvc/2048.model" }
    #dvc_config_template = { "path": f"{ROOT_DIR}/models/dvc/1024.model" }
    #dvc_config_template = { "path": f"{ROOT_DIR}/models/dvc/512.model" }
    #dvc_config_template = { "path": f"{ROOT_DIR}/models/dvc/256.model" }
    #qmap_coder = QmapModel(qmap_config_template)
    qmap_coder = None
    dvc_coder = DVCModel(dvc_config_template)

    ae_model = AEModel(qmap_coder, dvc_coder)
    ae_model.set_gop(8)
    # ae_model.set_gop(600) # for testing

    return ae_model

def generate_result_list(psnrs, bpps, lossvalue, qp, frame_shape):
    """
    psnrs, bpps: list of values
    lossvalue, qp: single value
    frame_shape: (width, height)
    Returns:
        list of <frame_id> <size in bytes> <psnr> <loss> <qp>
    """
    w, h = frame_shape
    res = []
    frame_id = 0
    for psnr, bpp in zip(psnrs, bpps):
        size_in_bytes = bpp * w * h / 8
        res.append([frame_id, size_in_bytes, float(psnr), lossvalue, qp])
        frame_id += 1
    return res

def debug(qp, use_mpeg=True):
    import dvc.net
    dvc.net.DEBUG_USE_MPEG = use_mpeg
    os.system("rm -f ../debug/codec/*")
    shape = (1280, 768)
    frames = read_video_into_frames(video_file, shape, 10)
    #frames = frames[-10:]
    ae_model = init_ae_model()

    #ae_model.set_quantization_param(qp)
    #psnrs, bpps = ae_model.debug_encode_video(frames, use_mpeg=use_mpeg)
    #psnrs, bpps = ae_model.encode_video(frames, use_mpeg=use_mpeg)
    psnrs, bpps = ae_model.encode_with_loss(frames, np.full(len(frames), 0.3))
    print(np.mean(psnrs), np.mean(bpps))

    #from dvc.dvc_model import mv_value_debugger
    #dbg1 = mv_value_debugger[qp].get(91).cpu()
    #dbg2 = ae_model.entropy_coder.get_debug_distribution_mv(91)
    #print(dbg1)
    #print(dbg2)
    #import pdb
    #pdb.set_trace()
    #for channel in range(128):
    #    dbg1 = mv_value_debugger[qp].get(channel).cpu()
    #    dbg2 = ae_model.entropy_coder.get_debug_distribution_mv(channel)
    #    dbg1 = dbg1 / torch.sum(dbg1)
    #    print(channel, torch.mean(torch.pow(dbg2-dbg1, 2)))

def debug_mpeg(qp):
    shape = (1280, 768)
    frames = read_video_into_frames(video_file, shape, 720)
    #frames = frames[-10:]
    mpeg_model = MPEGModel(gop=8)

    mpeg_model.set_qp(qp)
    psnrs, bpps = mpeg_model.encode_video(frames)
    print(np.mean(psnrs), np.mean(bpps))

class OptimizeConfig:
    def __init__(self, num_top_segs, num_copies, seg_size, debug):
        self.num_top_segs = num_top_segs
        self.num_copies = num_copies
        self.seg_size = seg_size #32
        self.debug = debug

class OptimizeStats:
    def __init__(self):
        self.top_seg_to_count = {}

    def record_top_seg(self, key):
        if key in self.top_seg_to_count:
            self.top_seg_to_count[key] = self.top_seg_to_count[key] + 1
        else:
            self.top_seg_to_count[key] = 1

    def print_popular_keys(self, map):
        queue = PriorityQueue()
        sum_val = 0
        for key in map:
            queue.put((-map[key], key))
            sum_val += map[key]
        top = 100
        for i in range(top):
            if queue.empty():
                break
            key = queue.get()[1]
            print(str(i)+"\t"+key+" : "+str(map[key])+
                    " ("+str(float(map[key]/sum_val))+")")

if __name__ == "__main__":

    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print_usage()

    video_file = sys.argv[1]
    outfile = sys.argv[2]
    mode = sys.argv[3]

    if sys.argv[4:]:
        dvc_path = sys.argv[4]
    else:
        dvc_path = None

    #for qp in [15, 21, 27, 33, 39]:
    #    debug_mpeg(qp)
    #exit(0)
    #debug(1, False)
    #debug(2, False)
    #debug(5, False)
    #debug(10, False)
    #exit(0)
    #debug(0)
    #debug(11)
    #debug(15)
    #debug(18)
    #debug(19)
    #debug(24)
    #debug(30)
    #debug(36)
    #debug(23)
    #debug(27)
    #debug(45)
    #debug(51)
    #debug(1.25)
    #debug(1.5)
    #debug(1)
    #debug(10) #debug(100)
    #exit(0)

    if mode != "mpeg" and mode != "ae":
        print_usage()

    ''' read the frames '''
    #shape = (1280, 768)
    shape = None
    frames = read_video_into_frames(video_file, shape, 720)
    frames = frames[0:60] # use the first 12 frames
    shape = frames[0].size
    print("Got {} frames".format(len(frames)))

    """ debug! """
    #ae_model = init_ae_model(dvc_path)
    #ae_model.set_debug_output_dir("/datamirror/yihua98/projects/autoencoder_testbed/debug/debug_loss")
    #psnrs, _ = ae_model.encode_with_loss(frames, np.full(len(frames), 0.75))

    ''' encode '''
    if mode == "ae":
        ae_model = init_ae_model(dvc_path)
        results = [] # result format: <frameid> <size in bytes> <psnr> <loss> <qp>
        for qp in [1]:
            ae_model.set_quantization_param(qp)

            ''' skip no-loss testing for now
            psnrs, bpps = ae_model.encode_video(frames)
            loss = 0
            lis = generate_result_list(psnrs, bpps, loss, qp, shape)
            results.extend(lis)
            print(qp, loss, np.mean(psnrs), np.mean(bpps))
            '''

            # 42.93029088240404 0.05805538862179487
            loss = 0
            losses = np.full(len(frames), loss)
            config = OptimizeConfig(0, 0, 32, False)
            psnrs, bpps, stats = ae_model.encode_with_loss_optimized_v2(frames, losses, config, True)
            psnrs = psnrs[1:]; bpps = bpps[1:]
            lis = generate_result_list(psnrs, bpps, loss, qp, shape)
            results.extend(lis)
            print(qp, loss, np.mean(psnrs), np.mean(bpps))

            # for loss in [0.15, 0.3, 0.46, 0.6, 0.75, 0.9]:
            for loss in [0.3, 0.6]:
                losses = np.full(len(frames), loss)
                # psnrs, _ = ae_model.encode_with_loss(frames, losses)
                # psnrs, bpps = ae_model.encode_with_loss(frames, losses, True)
                # psnrs, bpps = ae_model.encode_with_loss_optimized_with_gradient(frames, losses, True)
                # psnrs, bpps = ae_model.encode_with_loss_optimized_v2(frames, losses, config, True)
                # lis = generate_result_list(psnrs, bpps, loss, qp, shape)
                # results.extend(lis)
                # print(qp, loss, np.mean(psnrs), np.mean(bpps))

                # 37.71957544180063 0.06624849759615385
                config = OptimizeConfig(0, 0, 32, False)
                psnrs, bpps, stats = ae_model.encode_with_loss_optimized_v2(frames, losses, config, True)
                lis = generate_result_list(psnrs, bpps, loss, qp, shape)
                results.extend(lis)
                print(qp, loss, np.mean(psnrs), np.mean(bpps))

                # 40.49750878260686 0.06407502003205127
                # config = OptimizeConfig(2, 1, 32, False)
                # psnrs, bpps, stats = ae_model.encode_with_loss_optimized_v2(frames, losses, config, True)
                # lis = generate_result_list(psnrs, bpps, loss, qp, shape)
                # results.extend(lis)
                # print(qp, loss, np.mean(psnrs), np.mean(bpps))

                # 40.60519533890944 0.06401116786858975
                # (static) 40.427040760333725 0.06416516426282051
                # config = OptimizeConfig(3, 1, 32, False)
                # psnrs, bpps, stats = ae_model.encode_with_loss_optimized_v2(frames, losses, config, True)
                # lis = generate_result_list(psnrs, bpps, loss, qp, shape)
                # results.extend(lis)
                # print(qp, loss, np.mean(psnrs), np.mean(bpps))

                # 40.78689142373892 0.06396484375
                # (static) 40.43399165226863 0.06425530849358975
                # config = OptimizeConfig(4, 1, 32, False)
                psnrs, bpps, stats = ae_model.encode_with_loss_optimized_v2(frames, losses, config, True)
                lis = generate_result_list(psnrs, bpps, loss, qp, shape)
                results.extend(lis)
                print(qp, loss, np.mean(psnrs), np.mean(bpps))
                # stats.print_popular_keys(stats.top_seg_to_count)

                # 40.999282323397125 0.07073818108974358
                # (static) 41.227694217975326 0.07255358573717947
                # config = OptimizeConfig(8, 1, 32, False)
                # psnrs, bpps, stats = ae_model.encode_with_loss_optimized_v2(frames, losses, config, True)
                # lis = generate_result_list(psnrs, bpps, loss, qp, shape)
                # results.extend(lis)
                # print(qp, loss, np.mean(psnrs), np.mean(bpps))

                # 41.12442185328557 0.09582206530448721
                # config = OptimizeConfig(8, 2, 32, False)
                # psnrs, bpps, stats = ae_model.encode_with_loss_optimized_v2(frames, losses, config, True)
                # lis = generate_result_list(psnrs, bpps, loss, qp, shape)
                # results.extend(lis)
                # print(qp, loss, np.mean(psnrs), np.mean(bpps))

        freeze_psnr = ae_model.get_avg_freeze_psnr(frames)
        results.append([-1, -1, freeze_psnr, -1, -1])
        df = pd.DataFrame(results, columns=["frame_id", "size", "psnr", "loss", "qp"])
        df.to_csv(outfile, index=None)
        exit(0)

    if mode == "mpeg":
        ''' use P FRAME '''
        mpeg_model = MPEGModel(gop=8, use_265=True)
        ''' Only I frame '''
        #mpeg_model = MPEGModel(gop=1, only_P=False, use_265=False) # for I frames
        results = [] # result format: <frameid> <size_in_bytes> <psnr> <qp>
        for qp in range(10, 50, 8):
            mpeg_model.set_qp(qp)
            psnrs, bpps = mpeg_model.encode_video(frames)

            fid = 0
            for psnr, bpp in zip(psnrs, bpps):
                size_in_bytes = bpp * shape[0] * shape[1] / 8
                results.append([fid, size_in_bytes, float(psnr), qp])
                fid += 1

            print(qp, np.mean(psnrs), np.mean(bpps))
        freeze_psnr = mpeg_model.get_avg_freeze_psnr(frames)
        results.append([-1, -1, freeze_psnr, -1])
        df = pd.DataFrame(results, columns=["frame_id", "size", "psnr", "qp"])
        df.to_csv(outfile, index=None)
        exit(0)
