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
from torchvision.transforms.functional import *
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
from sessionconfig import *
from network import *
from codec import *
import random


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

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


class RateAdaptor:
    def __init__(self):
        pass

    def get_rate_of_frame(self, session_config):
        """
        TODO: This logic will be integrated with simulator to implement
        transport level rate adaptation
        """
        return session_config.get_static_target_bpp()


class Video:
    def __init__(self, frames):
        self.frames = frames

    def get_frame(self, frame_idx):
        return self.frames[frame_idx]

    def get_all_frames(self):
        return self.frames


class Session:
    def __init__(self, session_config):
        self.session_config = session_config

    def run(self):
        bpps = []
        psnrs = []
        session_config = self.session_config
        link = Link(session_config.get_static_loss())
        adaptor = RateAdaptor()
        codec = Codec()
        packetizer = Packetizer()
        frames = session_config.get_video().get_all_frames()

        for frame_idx, frame in enumerate(tqdm(frames)):
            target_rate = adaptor.get_rate_of_frame(session_config)
            eframe = codec.encode(frame_idx, target_rate, session_config)
            encoder_config = codec.encoder_config
            flow, bpp, eframe = packetizer.to_flow(eframe,
                                    encoder_config.get_model(),
                                    session_config)
            flow = link.send(flow, session_config)
            eframe = packetizer.from_flow(flow, eframe, session_config)
            decoded = codec.decode(eframe, frame_idx, session_config)
            if eframe.frame_type == "P": # I-frames are ignored for now
                bpps.append(bpp)
                psnr = METRIC_FUNC(to_tensor(frame), decoded)
                psnrs.append(psnr)
        return psnrs, bpps



if __name__ == "__main__":

    video_file = "../data/test_videos_old/-01cbva4erQ_000007_000017.mp4"
    shape = None
    frames = read_video_into_frames(video_file, shape, 720)
    frames = frames[0:20]
    print("Got {} frames".format(len(frames)))

    dvc_path = "/datamirror/autoencoder_dataset/snapshot/"
    dvc_path_256 = dvc_path + "256.model"
    dvc_path_512 = dvc_path + "512.model"
    dvc_path_1024 = dvc_path + "1024.model"
    dvc_path_2048 = dvc_path + "2048.model"
    dvc_path_4096 = dvc_path + "4096.model"
    model_list = [dvc_path_256, dvc_path_512, dvc_path_1024, dvc_path_2048, dvc_path_4096]

    # codec_manager is stateless, so can be reused!
    video = Video(frames)
    codec_manager = CodecManager(model_list)
    loss = 0.0
    target_bpp = 0.05
    static_model_id = 3
    for ref_frame_type in [REF_FRAME_GT, REF_FRAME_APPROX_DECODED, REF_FRAME_SYNCED_DECODED]:
        model_id = codec_manager.get_ae_id(static_model_id)
        session_config = SessionConfig(video, codec_manager,
                                       static_loss = loss,
                                       static_target_bpp = target_bpp,
                                       ref_frame_type = ref_frame_type,
                                       static_model_id = model_id)
        session = Session(session_config)
        psnrs, bpps = session.run()
        print(np.mean(psnrs), np.mean(bpps),
             np.count_nonzero(np.asarray(bpps) > target_bpp),
             session_config.print())

    for static_model_idx in [0, 2, 4]:
        model_id = codec_manager.get_ae_id(static_model_idx)
        session_config = SessionConfig(video, codec_manager,
                                       static_loss = loss,
                                       static_target_bpp = target_bpp,
                                       static_model_id = model_id)
        session = Session(session_config)
        psnrs, bpps = session.run()
        print(np.mean(psnrs), np.mean(bpps),
             np.count_nonzero(np.asarray(bpps) > target_bpp),
             session_config.print())

    for static_qp in [16, 24]:
        model_id = codec_manager.get_h26x_id(static_qp)
        session_config = SessionConfig(video, codec_manager,
                                       static_loss = loss,
                                       static_target_bpp = target_bpp,
                                       static_model_id = model_id)
        session = Session(session_config)
        psnrs, bpps = session.run()
        print(np.mean(psnrs), np.mean(bpps),
             np.count_nonzero(np.asarray(bpps) > target_bpp),
             session_config.print())
