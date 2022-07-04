import torchac
import os, sys
import zlib
import torch
import numpy as np
import subprocess as sp
import cv2
import time
import shlex
from PIL import Image, ImageFile, ImageFilter

def read_video_into_frames(video_path, nframes=1000):
    """
    Input:
        video_path: the path to the video
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
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        frames.append(img)

        if len(frames) >= nframes:
            break
    return frames

class TorchacCoder:
    def __init__(self):
        pass

    def entropy_encode(self, x: np.ndarray):
        """
        encode the input array 
        returns:
            bytestream: encoded bytestream
            cdf: the generated cdf
            size: encoded size in bytes
            minval: normalized minval (origin = decoded + minval)
        """
        x = x.flatten()
        length = x.size
    
        # get max and min for x
        maxval = max(x)
        minval = min(x)
    
        x = x - minval # now, xrange is [0, maxval - minval]
        maxval -= minval
        print(min(x), max(x))
    
        # compute CDF for x
        cdf = [0]
        for i in range(maxval + 1):
            cdf.append((x < i).sum() / length)
        cdf.append(1)
        orig_shape = len(cdf)
        cdf = np.tile(np.array(cdf), length).reshape(-1, orig_shape)
        
        x = torch.from_numpy(x).type(torch.int16)
        cdf = torch.from_numpy(cdf)

        # compress x
        bytestream = torchac.encode_float_cdf(cdf, x, check_input_bounds=True, needs_normalization=True)
        return bytestream, cdf, len(bytestream), minval
    
    def entropy_decode(self, bytestream, cdf, minval):
        sym_out = torchac.decode_float_cdf(cdf, bytestream, needs_normalization=True)
        sym_out += minval
        return sym_out

class GzipCoder:
    def __init__(self, dtype=np.int16):
        self.dtype = dtype
        pass

    def entropy_encode(self, x: np.ndarray, fast=True):
        if not fast and self.dtype == np.int8:
            newx = np.zeros_like(x)
            for idx, val in enumerate(x):
                if val < -128:
                    val = -128
                elif val > 127:
                    val = 127
                newx[idx] = val
            x = newx

        x = x.astype(self.dtype)
        bytestream = zlib.compress(x.tobytes(), level=9)
        size = len(bytestream)
        return bytestream, size

    def entropy_decode(self, bytestream):
        data = zlib.decompress(bytestream)
        code = np.frombuffer(data, dtype=self.dtype)
        return code

class MPEGCoder:
    def __init__(self):
        self.height = 256
        self.width = 256

        self.cmd = None
        self.qp = 21
        self.process = None

        val = np.random.random_integers(0, 10000)
        self.fenc = open(f"/tmp/MPEGCoder-fenc-{val}.mp4", "wb+")
        self.enc_len = 0
        self.fdec = open(f"/tmp/MPEGCoder-fdec-{val}.mp4", "wb+")
        self.dec_len = 0

        #self.base_list = \
        #    [0] * 16 + [1] * 8 + [2] * 8 + [3] * 8 + [4] * 8 + [5] * 4 + [6] * 4 + [7] * 4 + [8] * 4 + \
        #    list(range(9, 128, 2)) #+ [92, 94, 98, 102, 108, 116 ,126]
        #self.encode_list = \
        #    [0] * 20 + [1] * 8 + [2] * 8 + [3] * 8 + [4] * 6 + [5] * 4 + [6] * 4 + [7] * 4 + [8] * 2 + \
        #    list(range(9, 128, 2)) #+ [92, 94, 98, 102, 108, 116 ,126]
        #self.decode_list = np.digitize(np.arange(0, 128), self.base_list, True)[1:]

        #self.base_list = \
        #    [0] * 16 + [1] * 16 + [2] * 16 + [3] * 16 + [4] * 16 + [5] * 16 + list(np.repeat(np.arange(6, 16), 2))
        #self.encode_list = \
        #    [0] * 24 + [1] * 16 + [2] * 16 + [3] * 16 + [4] * 16 + [5] * 9 + list(np.repeat(np.arange(6, 16), 2))
        self.base_list = list(np.repeat(np.arange(0, 16), 8))
        self.encode_list = [0] * 12 + list(np.repeat(np.arange(1, 15), 8)) + [15] * 4
        self.decode_list = np.digitize(np.arange(0, 17), self.base_list, True)[1:]

        self.base_list = np.array(self.base_list)
        self.encode_list = np.array(self.encode_list)
        self.decode_list = np.array(self.decode_list)

        self.debug = False
        self.debug_input = None
        self.debug_inimg = None
        self.debug_outimg = None
        self.debug_output = None

        self.image_id = 1

    def enable_debug(self):
        self.debug = True

    def set_qp(self, qp):
        self.qp = qp

    def fit_code_size(self, sz):
        assert self.cmd is None, "Can only call fit_code_size() for once"
        assert sz % (self.width * 3) == 0, "size is not dividable!"
        self.height = int(sz // self.width // 3)
        self.cmd = f'/usr/bin/ffmpeg -y -s {self.width}x{self.height} -pixel_format bgr24 -r 25 -f rawvideo -i pipe:0 -vcodec libx264 -pix_fmt yuv420p -preset veryfast -tune zerolatency -crf {self.qp} -g 20 -sc_threshold 0 -loglevel debug -f rawvideo pipe:1'
        #self.cmd = f'/usr/bin/ffmpeg -y -s {self.width}x{self.height} -r 25 -f rawvideo -i pipe:0 -vcodec libx264 -crf {self.qp} -f rawvideo pipe:1'
        print("cmd is:", self.cmd)
        self.process = sp.Popen(shlex.split(self.cmd), stdin=sp.PIPE, stdout=self.fenc, stderr=sp.STDOUT)

    def scale_code(self, code, new_min, new_max):
        v_min, v_max = code.min(), code.max()
        code = (code - v_min) / (v_max - v_min) * (new_max - new_min) + new_min
        return code, v_min, v_max

    def scale_code_new(self, code):
        signs = np.sign(code)
        code = np.abs(code)
        code = signs * np.digitize(code, self.encode_list, True)

        vmin, vmax = code.min(), code.max()
        vmin, vmax = -128, 127
        code = code - vmin

        gap = vmax - vmin
        code = code.clip(0, 255)
        gap = 256
        scale = np.floor(256 / gap)
        code = code * scale

        #temp = 8
        #scale = 1
        #code = code * temp + (code != 0).astype("int") * (temp // 2)

        #vmin = code.min()
        #code -= vmin

        return code, vmin, scale

    def rescale_code_new(self, code, vmin, scale):
        temp = 8
        code = code / scale
        code = code + vmin

        signs = np.sign(code)
        code = np.abs(code)
        code = signs * np.digitize(code, self.decode_list, True)
        #code = code - temp // 2
        #code = np.clip(code, a_min=0, a_max = 256)
        #code = signs * code / temp
        return code

    def code_to_image(self, code: np.ndarray):
        """
        Input:
            code: numpy array of the tensor
        Output:
            image: the PIL image scaled to 0-255 for ffmpeg to compress
            vmin: the min value in the original code
            vmax: the max value in the original code
        """
        #code, vmin, vmax = self.scale_code(code, 0, 255)
        code, vmin, vmax = self.scale_code_new(code)

        code = code.reshape((self.height, self.width, 3)).astype("uint8")
        img = Image.fromarray(code)
        print(img.size)
        return img, vmin, vmax


    def image_to_code(self, image:Image, vmin, vmax):
        """
        Input:
            image: the encoded image
            vmin: the min value in the original code
            vmax: the max value in the original code
        Output:
            code: restored code
        """
        code = np.array(image)
        #code, _, _ = self.scale_code(code, vmin, vmax)
        code = self.rescale_code_new(code, vmin, vmax)
        return code.flatten()

    def debug_entropy_encode(self, np_code: np.ndarray):
        """
        Input:
            np_code: flattened numpy array
        Output:
            label: the output label
            size: the estimated size in bytes
            vmin, vmax: used for decode
        """
        img, vmin, vmax = self.code_to_image(np_code)
        ret_id = self.image_id

        path = "/datamirror/yihua98/projects/autoencoder_testbed/debug/codec/"
        pngname = "{}/input-{:02}.png".format(path, self.image_id)
        videoname = "{}/out.mp4".format(path)
        jpgname = "{}/test-{:02}.jpg".format(path, self.image_id)

        if self.image_id == 1:
            os.system("rm -f {}".format(videoname))
            os.system("rm -f {}/input*".format(path))
            os.system("rm -f {}/output*".format(path))

        ''' get previous size '''
        prev_size = 0
        if os.path.exists(videoname):
            prev_size = os.path.getsize(videoname)
        
        ''' encode '''
        img.save(pngname)
        self.image_id += 1
        print("using qp =", self.qp)
        os.system("ffmpeg -y -i {}/input-%02d.png -c:v libx264 -qp {} -vf fps=25 {} 2>/dev/null".format(path, self.qp, videoname))
        #os.system(f"convert {pngname} -quality 95 {jpgname}") # jpg compression

        ''' get estimated output size '''
        now_size = os.path.getsize(videoname)
        while now_size == prev_size:
            time.sleep(0.1)
            now_size = os.path.getsize(videoname)
        est_size = now_size - prev_size
        #est_size = os.path.getsize(jpgname) # jpg compression

        if self.debug:
            self.debug_input = np_code
            self.debug_inimg = img

        return ret_id, est_size, vmin, vmax

    def debug_entropy_decode(self, label, vmin, vmax):
        """
        Input:
            code: the compressed bytestream
            vmin: the min value in the original code
            vmax: the max value in the original code
        Output:
            code: the decoded numpy array
        """
        path = "/datamirror/yihua98/projects/autoencoder_testbed/debug/codec/"
        pngname = "{}/output-{:02}.png".format(path, label)
        videoname = "{}/out.mp4".format(path)
        jpgname = "{}/test-{:02}.jpg".format(path, label)

        ''' read the video and get the image based on label '''
        os.system("ffmpeg -y -r 1 -i {} {}/output-%02d.png 2>/dev/null".format(videoname, path))
        img = Image.open(pngname)
        #img = Image.open(jpgname) # jpg compression

        code = self.image_to_code(img, vmin, vmax)
        if self.debug:
            self.debug_outimg = img
            self.debug_output = code
        return code

    def entropy_encode(self, np_code: np.ndarray):
        """
        Input:
            np_code: flattened numpy array
        Output:
            code: the bytestream after compression
            vmin: the min value in the original code
            vmax: the max value in the original code
        """
        img, vmin, vmax = self.code_to_image(np_code)
        #return np.array(img).flatten(), vmin, vmax # debug

        ''' original encoding process '''
        self.process.stdin.write(np.array(img).tobytes())
        while self.fenc.tell() == self.enc_len:
            time.sleep(0.1)

        updated_len = self.fenc.tell()
        self.fenc.seek(self.enc_len)
        arr = self.fenc.read()
        self.enc_len = updated_len

        if self.debug:
            self.debug_input = np_code
            self.debug_inimg = img
        return arr, vmin, vmax

    def entropy_decode(self, code, vmin, vmax):
        """
        Input:
            code: the compressed bytestream
            vmin: the min value in the original code
            vmax: the max value in the original code
        Output:
            code: the decoded numpy array
        """
        #import pdb
        #pdb.set_trace()
        self.fdec.write(code)
        self.fdec.flush()
        fname = self.fdec.name
        frames = read_video_into_frames(fname)
        img = frames[-1]
        code = self.image_to_code(img, vmin, vmax)
        #code = self.image_to_code(code, vmin, vmax) # debug
        if self.debug:
            self.debug_outimg = img
            self.debug_output = code
        return code

    def get_debug_info(self):
        ''' dump 2 images '''
        self.debug_inimg.save("/tmp/inimg.png")
        self.debug_outimg.save("/tmp/outimg.png")
        return self.debug_input, np.array(self.debug_inimg).flatten(), np.array(self.debug_outimg).flatten(), self.debug_output

class MPEGCoderWrapper:
    def __init__(self):
        self.impl = MPEGCoder()

    def set_qp(self, qp):
        self.impl.set_qp(qp)

    def fit_code_size(self, sz):
        self.impl.fit_code_size(sz)

    def entropy_encode(self, x, shapex, shapey, z):
        """
        x: flattened torch array
        returns bs, tot_size, vmin, vmax
        """
        nparr = x.cpu().detach().numpy().flatten()
        bs, vmin, vmax = self.impl.entropy_encode(nparr)
        return bs, len(bs), vmin, vmax
    
    def entropy_decode(self, bs, vmin, vmax):
        """
        returns torch tensor
        """
        v = self.impl.entropy_decode(bs, vmin, vmax).flatten()
        return torch.from_numpy(v)

    def debug_entropy_encode(self, x, shapex, shapey, z):
        nparr = x.cpu().detach().numpy().flatten()
        bs, sz, vmin, vmax = self.impl.debug_entropy_encode(nparr)
        return bs, sz, vmin, vmax

    def debug_entropy_decode(self, x, vmin, vmax):
        v = self.impl.debug_entropy_decode(x, vmin, vmax)
        return torch.from_numpy(v)
