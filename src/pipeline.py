import numpy as np
import time
import torch
from qmap.qmap_model import QmapModel
from dvc.dvc_model import DVCModel
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from PIL import Image, ImageFile, ImageFilter

def to_pil_image(out: torch.tensor):
    fname = "/tmp/0fias832.png"
    save_image(out, fname)
    return Image.open(fname).convert("RGB")

def apply_loss(code, loss_rate, blocksize = 100):
    leng = len(code)
    nblocks = leng // blocksize
    #if nblocks * loss_rate < 1:
    #    print("Warning: Number of blocks ({}) are very small! Loss rate = {}".format(nblocks, loss_rate))

    newcode = np.array([])
    for i in range(0, leng, blocksize):
        dice = np.random.uniform(0, 1)
        codelen = min(i + blocksize, leng) - i
        if dice <= loss_rate:
            newcode = np.concatenate((newcode, np.zeros(codelen)), axis=None)
        else:
            newcode = np.concatenate((newcode, code[i:i+codelen]), axis=None)
    return newcode

def apply_loss_v2(code, loss_rate, blocksize = 100):
    leng = len(code)
    nblocks = leng // blocksize

    rnd = np.random.uniform(0, 1, nblocks);
    rnd = (rnd > loss_rate).astype("int")
    rnd = np.repeat(rnd, blocksize)
    rnd = rnd[:leng]

    ret = code * rnd
    return ret

class EncodedFrame:
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
        self.code = apply_loss_v2(self.code, loss_ratio, blocksize)


class FinalEncoder:
    def __init__(self, qmap_config, dvc_config, qmap_coder=None, dvc_coder=None):
        self.qmap_conf = qmap_config
        self.dvc_conf = dvc_config

        if qmap_coder is None:
            self.qmap_coder = QmapModel(self.qmap_conf)
        else:
            self.qmap_coder = qmap_coder 

        if dvc_coder is None:
            self.dvc_coder = DVCModel(self.dvc_conf)
        else:
            self.dvc_coder = dvc_coder

        self.ref_frame = None # should be torch.tensor() on the device (easy to use for the models)
        self.encode_time_list = []
        self.frame_counter = 0
        self.ref_id = 0

    def _update_reference(self, code, frame_type, shapex, shapey):
        if frame_type == "I":
            out = self.qmap_coder.decode(code, shapex, shapey)
        elif frame_type == "P":
            if self.ref_frame is None:
                raise RuntimeError("Internal error: no reference frame when try to decode a P frame")
            out = self.dvc_coder.decode(code, self.ref_frame, shapex, shapey)
        else:
            raise RuntimeError("Internal error: unknown frame_type!")
        self.ref_frame = out

    def _record_time(self, start):
        newt = time.time()
        self.encode_time_list.append(newt - start)

    def update_reference(self, new_reference_id, eframe_hat: EncodedFrame):
        """
        This function is used to sync-up the reference frame between encoder and decoder
        Parameter:
            new_reference_id: the ID of reference frame used at receiver side
            eframe_hat: the encoded frame after loss
        """
        self._update_reference(torch.from_numpy(eframe_hat.code), eframe_hat.frame_type, eframe_hat.shapex, eframe_hat.shapey)
        self.ref_id = new_reference_id

    def set_Iframe(self, frame, ref_id=None):
        """
        frame: is the PIL image
        """
        if ref_id is None:
            ref_id = self.ref_id + 1
        self.ref_id = ref_id
        self.ref_frame = to_tensor(frame)
        return ref_id

    def debug_decode(self, eframe:EncodedFrame, frame_type='P'):
        if frame_type != "P":
            raise RuntimeError("Debug mode only support P frames!")
        code = torch.from_numpy(eframe.code)
        out = self.dvc_coder.decode(code, self.ref_frame, eframe.shapex, eframe.shapey)
        return out

    #def encode(self, frame, frame_type, update_reference=True):
    def encode(self, frame, frame_type):
        """
        Parameter:
            frame: the input image, either PIL image or numpy ndarray of H,W,C in range [0, 255] (it will be fed through to_tensor)
            frame_type: 'I' or 'P'
        Returns:
            eframe: the encoded frame
            ref_number: the number of reference frame
        """
        self.frame_counter += 1
        start_time = time.time()
        frame = to_tensor(frame)
        if frame_type == 'I':
            code, shapex, shapey = self.qmap_coder.encode(frame)
            np_code = code.cpu().detach().numpy().flatten()

            self.ref_id = self.frame_counter
            self._update_reference(code, "I", shapex, shapey)
            eframe = EncodedFrame(np_code, shapex, shapey, "I", self.frame_counter)
            self._record_time(start_time)
            return eframe, None # I frame does not depend on other frames

        elif frame_type == 'P':
            if self.ref_frame is None:
                raise RuntimeError("Cannot encode a P frame without any reference frame!")
            code, shapex, shapey = self.dvc_coder.encode(frame, self.ref_frame)
            np_code = code.cpu().detach().numpy().flatten()
            #if update_reference:
            #    self._update_reference(code, "P", shapex, shapey) 
            self._record_time(start_time)
            return EncodedFrame(np_code, shapex, shapey, "P", self.frame_counter), self.ref_id
        else:
            raise RuntimeError("Unknown frame type {}".format(frame_type))

    def get_fps(self):
        return 1/np.mean(self.encode_time_list)

class FinalDecoder:
    def __init__(self, qmap_config, dvc_config, qmap_coder = None, dvc_coder=None):
        self.qmap_conf = qmap_config
        self.dvc_conf = dvc_config

        if qmap_coder is None:
            self.qmap_coder = QmapModel(self.qmap_conf)
        else:
            self.qmap_coder = qmap_coder 

        if dvc_coder is None:
            self.dvc_coder = DVCModel(self.dvc_conf)
        else:
            self.dvc_coder = dvc_coder

        #self.ref_frame = None # should be torch.tensor() on the device (easy to use for the models)
        self.decode_time_list = []
        self.ref_list = {} # <key, val> = <ID, decoded frame>

        self.latest_rid = 0
        self.latest_eframe = None

    def _record_time(self, start):
        newt = time.time()
        self.decode_time_list.append(newt - start)

    def _update_ref_list(self, current_rid):
        if current_rid is None:
            return
        keys = list(self.ref_list.keys())
        for key in keys:
            if key < current_rid:
                self.ref_list.pop(key)

    def _new_reference(self, rid, ref_frame):
        self.latest_rid = rid
        self.ref_list[self.latest_rid] = ref_frame

    def decode(self, eframe: EncodedFrame, reference_id):
        """
        Parameter:
            code: 1-D numpy array on cpu
            frame_type: I or P
            shapex, shapey: the shapes used for decode
        Returns:
            out: pytorch tensor (can be on GPU) with size C, H, W
        """
        start_time = time.time()
        code = torch.from_numpy(eframe.code)
        frame_type = eframe.frame_type
        shapex, shapey = eframe.shapex, eframe.shapey

        if frame_type == "I":
            out = self.qmap_coder.decode(code, shapex, shapey)
            #self.ref_frame = out # always update reference on I frame
        elif frame_type == "P":
            # use the correct reference frame
            if reference_id not in self.ref_list:
                raise RuntimeError("Cannot decode a P frame without any reference frame!")
            ref_frame = self.ref_list[reference_id]

            # decode
            out = self.dvc_coder.decode(code, ref_frame, shapex, shapey)
        else:
            raise RuntimeError("Unknown frame type {}".format(frame_type))

        # update the reference list and add new decoded frame into it
        self.latest_eframe = eframe
        self._new_reference(eframe.frame_id, out)
        self._update_ref_list(reference_id)

        temp = out.cpu().detach().numpy()
        self._record_time(start_time)
        return out
        #return to_pil_image(out) # to_pil_image is not working properly!

    def set_Iframe(self, frame, ref_id):
        """
        frame: is the PIL image
        """
        self.latest_rid = ref_id
        self._new_reference(ref_id, to_tensor(frame))
        self._update_ref_list(ref_id)

    def feedback(self):
        return self.latest_rid, self.latest_eframe

    def get_fps(self):
        return 1/np.mean(self.decode_time_list)

class DebugCoderPFrame:
    def __init__(self, dvc_coder, init_ref_frame):
        """
        init_ref_frame: PIL image
        """
        self.dvc_coder = dvc_coder
        self.frame_counter = 0
        self.set_ref_frame_pil(init_ref_frame)

    def encode(self, frame):
        """
        input:
            frame: PIL image

        returns:
            eframe: encoded frame
        """
        frame = to_tensor(frame)
        code, shapex, shapey = self.dvc_coder.encode(frame, self.ref_frame)
        np_code = code.cpu().detach().numpy().flatten()
        return EncodedFrame(np_code, shapex, shapey, "P", self.frame_counter)

    def decode(self, eframe: EncodedFrame):
        """
        input: 
            eframe; encoded frame
        returns:
            frame_hat: decoded frame, in torch.Tensor
        """
        code = torch.from_numpy(eframe.code)
        ref_frame = self.ref_frame
        shapex = eframe.shapex
        shapey = eframe.shapey
        out = self.dvc_coder.decode(code, ref_frame, shapex, shapey)
        return out

    def set_ref_frame_pil(self, frame):
        """
        input:
            frame: the PIL image
        """
        frame = to_tensor(frame)
        self.ref_frame = frame

    def set_ref_frame_np(self, frame):
        self.ref_frame = torch.from_numpy(frame)

    def set_ref_frame_tensor(self, frame):
        self.ref_frame = frame

