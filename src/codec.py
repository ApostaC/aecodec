import torch
import numpy as np
from sessionconfig import *
from torchvision.transforms.functional import *
from qmap.qmap_model import QmapModel
from dvc.dvc_model import DVCModel, DVCEntropyCoder
import cv2
import os, sys
import subprocess as sp
import io
import shlex

class EncodedFrame:
    """
    code is torch.tensor
    shapx is shape of encoded movitions
    shapy is shape of encoded residuals
    frame is the orginal frame torch.tensor (ONLY TO BE USED WHEN DEBUGGING)
    """
    def __init__(self, code, shapex, shapey, frame_type, frame = None):
        self.code = code
        self.shapex = shapex
        self.shapey = shapey
        self.frame_type = frame_type
        self.frame = frame

    def np_code(self):
        """
        return the code in flattened numpy array
        """
        return self.code.cpu().detach().numpy().flatten()

class AEModel:
    def __init__(self, qmap_coder: QmapModel, dvc_coder: DVCModel, only_P=True):
        self.qmap_coder = qmap_coder
        self.dvc_coder = dvc_coder
        self.entropy_coder = DVCEntropyCoder(dvc_coder)

    def encode_frame(self, frame, ref_frame, isIframe, session_config):
        """
        Input:
            tframe: torch.tensor of the current frame
            ref_frame: torch.tensor of the ref frame
        Output:
            eframe: encoded frame, code is torch.tensor on GPU
            bpp: estimated bpp of this frame
        Note:
            I-frame encoding is not implemented (it uses dvc instead)
        """
        if isIframe:
            # creating a dummy I-frame
            code, shapex, shapey = self.dvc_coder.encode(frame, ref_frame)
            eframe = EncodedFrame(code, shapex, shapey, "I", frame = frame)
            return eframe
        else:
            assert ref_frame is not None
            code, shapex, shapey = self.dvc_coder.encode(frame, ref_frame)
            eframe = EncodedFrame(code, shapex, shapey, "P", frame = frame)
            return eframe

    def decode_frame(self, eframe:EncodedFrame, ref_frame, session_config):
        """
        Input:
            eframe: the encoded frame (EncodedFrame object)
            ref_frame: reference frame (torch.tensor)
        Output:
            frame: the decoded frame in torch.tensor (3,h,w) on GPU, which
            can be used as ref frame
        Note:
            I-frame encoding is not implemented (instead, this should never
            be called for now)
        """
        if eframe.frame_type == "I":
            out = self.qmap_coder.decode(eframe.code, eframe.shapex, eframe.shapey)
            return out
        else:
            assert ref_frame is not None
            out = self.dvc_coder.decode(eframe.code, ref_frame, eframe.shapex, eframe.shapey)
            return out

    def get_size_in_bytes(self, eframe:EncodedFrame, session_config):
        zz = self.dvc_coder.encode_z(eframe.code[np.prod(eframe.shapex):].to(self.dvc_coder.device), eframe.shapey)
        zz = torch.round(zz)
        bs, tot_size = self.entropy_coder.entropy_encode(eframe.code,
                                eframe.shapex, eframe.shapey, zz)
        return tot_size
        # w, h = frame.size
        # w, h = session_config.get_frame_size()
        # bpp = tot_size * 8 / (w * h)
        # return bpp

class MPEGModel:
    def __init__(self):
        self.output_filename = "output.mp4"

    def encode_frame(self, frame, ref_frame, qp, isIframe, session_config):
        """
        Input:
            tframe: torch.tensor of the current frame
            ref_frame: torch.tensor of the ref frame
        Output:
            eframe: an empty encoded frame
            bpp: real ffmpeg encoded frame size
        """
        np_frame = to_pil_image(frame)
        np_ref_frame = to_pil_image(ref_frame)
        np_ref_frame.save("frame-001.png", format="png")
        np_frame.save("frame-002.png", format="png")
        cmd = f'/usr/bin/ffmpeg -y -i frame-%03d.png -vcodec libx265 -pix_fmt yuv420p -tune zerolatency -x265-params "qp={qp}:keyint=8:bframes=0:keyint=2:min-keyint=2" -loglevel fatal {self.output_filename}'
        process = sp.Popen(shlex.split(cmd), stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT)
        process.stdin.close()
        process.wait()
        process.terminate()
        if isIframe:
            eframe = EncodedFrame(None, 0, 0, "I", frame = frame)
        else:
            eframe = EncodedFrame(None, 0, 0, "P", frame = frame)
        return eframe

    def decode_frame(self, eframe:EncodedFrame, ref_frame, session_config):
        clip = []
        cap = cv2.VideoCapture(self.output_filename)
        while(cap.isOpened()):
            ret, img = cap.read()
            if ret != True:break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            clip.append(to_tensor(img))
            # to_pil_image(to_tensor(img)).save("frame-decoded-001.png", format="png")
        return clip[1]

    def get_size_in_bytes(self, eframe:EncodedFrame, session_config):
        bpp_res = sp.check_output(f'ffprobe -show_frames {self.output_filename} | grep "pkt_size\|pict_type" | grep "=P" -B 1 | grep "pkt_size"',
                                        stderr=open("/dev/null","w"), shell=True, encoding="utf-8")
        temp = bpp_res.split("\n")
        temp.remove("")
        sizes = list(map(lambda s: int(s.split("=")[1]), temp))
        # w, h = session_config.get_frame_size()
        # bpps = list(map(lambda B: B * 8 / (h * w), sizes))
        size = sizes[0]
        return size

class CodecManager:
    def __init__(self, model_list):
        self.model_paths = model_list
        self.loaded_id = None
        self.loaded_ae_model = None

    def get_ae_id(self, id):
        return KEYWORD_AE + "-" + str(id)

    def get_h26x_id(self, qp):
        return KEYWORD_26X + "-" + str(qp)

    def get_codec_config(self, model_id):
        if model_id.startswith(KEYWORD_AE):
            if self.loaded_id != model_id:
                id = int(model_id[len(KEYWORD_AE + "-"):])
                self.loaded_ae_model = init_ae_model(self.model_paths[id])
                self.loaded_id = model_id
            return create_ae_config(self.loaded_ae_model)
        elif model_id.startswith(KEYWORD_26X):
            qp = int(model_id[len(KEYWORD_26X + "-"):])
            return create_h26x_config(qp)
        return None

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
    qmap_coder = None
    dvc_coder = DVCModel(dvc_config_template)
    ae_model = AEModel(qmap_coder, dvc_coder)

    return ae_model

class CodecConfig:
    """
    type: either KEYWORD_AE or KEYWORD_26X
    model: AEModel if KEYWORD_AE or MPEGModel if KEYWORD_26X
    qp: used only in MPEGModel
    """
    def __init__(self, type, model, qp):
        self.type = type
        self.model = model
        self.qp = qp

    def get_model(self):
        return self.model

    def is_ae(self):
        return self.type == KEYWORD_AE

    def is_h26x(self):
        return self.type == KEYWORD_26X

    def encode_frame(self, tframe, ref_frame, isIframe, session_config):
        if self.is_ae(): # Autoencoder
            eframe = self.model.encode_frame(tframe,
                                ref_frame, isIframe, session_config)
            return eframe
        else: # H26x
            eframe = self.model.encode_frame(tframe,
                                ref_frame, self.qp, isIframe, session_config)
            return eframe

    def decode_frame(self, tframe, ref_frame, session_config):
        if self.is_ae():
            return self.model.decode_frame(tframe,
                                    ref_frame, session_config)
        else:
            return self.model.decode_frame(tframe,
                                    ref_frame, session_config)

def create_ae_config(model):
    return CodecConfig(KEYWORD_AE, model, qp = 1)

def create_h26x_config(qp):
    return CodecConfig(KEYWORD_26X, MPEGModel(), qp)

class Codec:
    def __init__(self):
        self.gop = 8
        self.ref_frame_encoder = None
        self.ref_frame_decoder = None
        self.encoder_config = None
        self.decoder_config = None

    def set_gop(self, gop):
        self.gop = gop

    def select_configs(self, target_rate, tframe, ref_frame, session_config):
        """
        TODO: Write the rate control logic here.
        """
        codec_config = session_config.get_static_codec_config()
        if codec_config == None:
            model = session_config.get_codec_manager().get_ae_id(3)
            return create_ae_config(model), create_ae_config(model)
        else:
            return codec_config, codec_config

    def select_ref_frame(self, tframe, eframe, session_config):
        """
        TODO: Write reference frame selection logic here.
        Input:
            tframe: original current frame
            eframe: coded
        Output:
            next frame's reference
            Case REF_FRAME_GT:
                Use the frame itself
            Case REF_FRAME_APPROX_DECODED:
                Use decoded frame (under 0% loss)
            Case REF_FRAME_SYNCED_DECODED:
                Use decoder's current frame output (right now, its
                unreaslistic. will need to add a lag between the encoder
                state and decoder state)
        """
        if session_config.get_ref_frame_type() == REF_FRAME_GT:
            return tframe
        elif session_config.get_ref_frame_type() == REF_FRAME_APPROX_DECODED:
            return self.encoder_config.decode_frame(eframe,
                                                    self.ref_frame_encoder,
                                                    session_config)
        elif session_config.get_ref_frame_type() == REF_FRAME_SYNCED_DECODED:
            if self.ref_frame_decoder != None:
                return self.ref_frame_decoder # NOTE is unrealistic!
            else:
                return None

    def encode(self, frame_idx, target_rate, session_config):
        codec_manager = session_config.get_codec_manager()
        video = session_config.get_video()
        frame = video.get_frame(frame_idx)
        tframe = to_tensor(frame)
        self.encoder_config, self.decoder_config = \
                            self.select_configs(target_rate,
                                                tframe, self.ref_frame_encoder,
                                                session_config)
        isIframe = (frame_idx % self.gop == 0)
        if isIframe:
            eframe = self.encoder_config.encode_frame(tframe,
                            tframe, isIframe, session_config)
            self.ref_frame_encoder = tframe
            return eframe
        else:
            eframe = self.encoder_config.encode_frame(tframe,
                            self.ref_frame_encoder, isIframe, session_config)
            if self.encoder_config.is_ae():
                self.ref_frame_encoder = self.select_ref_frame(\
                                            tframe, eframe, session_config)
            else:
                self.ref_frame_encoder = tframe
        return eframe

    def decode(self, eframe, frame_idx, session_config):
        codec_manager = session_config.get_codec_manager()
        video = session_config.get_video()
        reference_frame = None
        isIframe = (eframe.frame_type == "I")
        if isIframe:
            self.ref_frame_decoder = to_tensor(video.get_frame(frame_idx))
            return eframe.frame
        else:
            decoded = self.decoder_config.decode_frame(eframe,
                        self.ref_frame_decoder, session_config)
            self.ref_frame_decoder = decoded
        return decoded
