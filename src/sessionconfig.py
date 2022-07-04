import torch
import numpy as np

KEYWORD_AE = "ae"
KEYWORD_26X = "h26x"

PACKETIER_LINEAR = 1
PACKETIER_INTERLEAVED = 2

REF_FRAME_GT = 1
REF_FRAME_APPROX_DECODED = 2
REF_FRAME_SYNCED_DECODED = 3

class SessionConfig:
    def __init__(self, video, codec_manager,
                 static_loss = 0.1,
                 static_target_bpp = 0.05,
                 ref_frame_type = REF_FRAME_APPROX_DECODED,
                 static_model_id = None):
        self.video = video
        self.w, self.h = self.video.get_frame(0).size
        self.codec_manager = codec_manager
        self.static_loss = static_loss # < 0: dynamic loss
        self.static_target_bpp = static_target_bpp # < 0: dynamic bitrate
        self.ref_frame_type = ref_frame_type
        self.static_model_id = static_model_id # None: dynamic model

    def get_video(self):
        return self.video

    def get_frame_size(self):
        return self.w, self.h

    def get_codec_manager(self):
        return self.codec_manager

    def get_static_loss(self):
        return self.static_loss

    def get_static_target_bpp(self):
        return self.static_target_bpp

    def get_ref_frame_type(self):
        return self.ref_frame_type

    def get_static_model_id(self):
        return self.static_model_id

    def get_static_codec_config(self):
        if self.static_model_id == None:
            return None
        else:
            return self.codec_manager.get_codec_config(\
                    self.static_model_id)

    def print(self):
        output = "[Config: "
        num_frames = len(self.get_video().get_all_frames())
        output += "#frames: "+str(num_frames)+", "
        output += "Loss: "+str(self.get_static_loss())+", "
        output += "Bpp: "+str(self.get_static_target_bpp())+", "
        output += "Model: "+str(self.get_static_model_id())+", "
        output += "Ref_frame: "+str(self.get_ref_frame_type())+", "
        output += "]"
        return output
