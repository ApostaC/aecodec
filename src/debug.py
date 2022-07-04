import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import time
import os, sys
import cv2
from PIL import Image, ImageFile, ImageFilter
import torch
import subprocess as sp
import shlex
from utils.entropy_coder import MPEGCoder

DEBUG_COUNTER = 0

def plot_pdf_tensor(vec, filename, remove_zero=True):
    """
    vec: torch tensor
    filename: output filename
    """
    vec = vec[vec!=0]
    values, counts = torch.unique(vec, return_counts=True)
    values = values.cpu().detach().numpy()
    counts = counts.cpu().detach().numpy()

    fout = PdfPages(filename)
    fig = plt.figure()
    plt.bar(values, counts)
    fout.savefig(fig)

    fig = plt.figure()
    plt.bar(values, counts)
    plt.yscale("log")
    fout.savefig(fig)
    fout.close()

def save_tensor_to_csv(vec, csv):
    arr = vec.cpu().detach().numpy().flatten()
    df = pd.DataFrame()
    df["values"] = arr
    df.to_csv(csv, index=False)

def save_tensors_to_csv(vecs, csv):
    df = pd.DataFrame()
    for idx, vec in enumerate(vecs):
        arr = vec.cpu().detach().numpy().flatten()
        df[f"value{idx}"] = arr
    df.to_csv(csv, index=False)

def save_np_to_csv(vec, csv):
    df = pd.DataFrame()
    df["values"] = vec 
    df.to_csv(csv, index=False)


def dump_tensor_to_image(tensor, filename):
    ''' scaling '''
    v_min, v_max = tensor.min(), tensor.max()
    new_min, new_max = 0, 255
    tensor = (tensor - v_min)/(v_max - v_min)*(new_max - new_min) + new_min

    ''' dump to image and save '''
    nparr = tensor.view(256, -1, 3).cpu().detach().numpy().astype("uint8")
    print(nparr.shape)
    img = Image.fromarray(nparr)
    img.save(filename)


def read_video_into_frames(video_path, frame_size, nframes=1000):
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

        img = Image.fromarray(img)
        img = img.resize(frame_size)
        frames.append(img)

        if len(frames) >= nframes:
            break
    return frames


def test_ffmpeg():
    #width, height = 512, 512
    #frames = read_video_into_frames("/datamirror/yihua98/projects/autoencoder_testbed/data/videos/pv724.mp4", (width, height), 10)

    dir="/datamirror/yihua98/projects/autoencoder_testbed/debug/pngs/"
    files=os.listdir(dir)
    frames = []
    for fname in sorted(files):
        img = Image.open(dir + fname)
        frames.append(img)
    width, height = img.size
    print(f"Got {len(frames)} images, resolution is {width}x{height}")

    cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -r 25 -f rawvideo -i pipe:0 -vcodec libx264 -pix_fmt yuv420p -preset veryfast -tune zerolatency -qp 9 -g 20 -sc_threshold 0 -loglevel debug -f rawvideo pipe:1'


    fout = open("/tmp/yihua.mp4", "wb+")

    process = sp.Popen(shlex.split(cmd), stdin=sp.PIPE, stdout=fout, stderr=sys.stdout)
    tot_len = 0
    for idx, img in enumerate(frames):
        print("IMG:", idx)
        process.stdin.write(np.array(img).tobytes())
        print("frame size =", fout.tell() - tot_len)
        tot_len = fout.tell()


    process.stdin.close()
    process.wait()
    #print("Output:", output)
    #output = open("/tmp/yihua.mp4","rb").read()
    print("total len =", tot_len)
    print("finished!")

def ffmpeg_encode_one(image):
    """
    image: the pil image
    returns:
        bs: the byte stream
        avg: avg pixel values used in ref construction
    """
    avg = int(np.mean(image))
    ref_arr = np.full_like(np.array(image), avg, dtype=np.uint8)
    ref_img = Image.fromarray(ref_arr)

    width, height = image.size
    cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -r 25 -f rawvideo -i pipe:0 -vcodec libx264 -pix_fmt yuv420p -preset veryfast -tune zerolatency -qp 9 -g 20 -sc_threshold 0 -loglevel debug -f rawvideo pipe:1'
    fout = open("/tmp/yihua.mp4", "wb+")
    process = sp.Popen(shlex.split(cmd), stdin=sp.PIPE, stdout=fout, stderr=sys.stdout)

    process.stdin.write(np.array(ref_img).tobytes())
    base_len = 0
    while base_len == 0:
        time.sleep(0.1)
        base_len = fout.tell()
    process.stdin.write(np.array(image).tobytes())


    process.stdin.close()
    process.wait()
    size = fout.tell() - base_len
    print("Compressed size is", size)
    fout.seek(base_len)
    arr = fout.read()
    return arr, avg


def debug_codec():
    img = Image.open("../data/pv724/0.png")
    img = img.resize((256, 256))
    SIZE = np.array(img).size
    codec = MPEGCoder()
    codec.fit_code_size(SIZE)
    codec.set_qp(0)

    code = codec.image_to_code(img, -128, 1)
    codec.enable_debug()

    code = code.flatten()
    bs, a, b = codec.entropy_encode(code)
    code2 = codec.entropy_decode(bs, a, b)
    codec.get_debug_info()

    def test_one(codec: MPEGCoder, code, noise):
        code = code.flatten()
        t1, a, b = codec.code_to_image(code)
        t1 = np.array(t1)
        noise_t1 = t1.copy().flatten()
        noises = np.arange(noise, -noise, -1)
        for idx, val in enumerate(noise_t1):
            noise_t1[idx] = val + noises[idx % noises.size]

        t2 = codec.image_to_code(noise_t1, a, b).flatten()
        for i in range(code.size):
            orig_value = code[i]
            image_value = t1.flatten()[i]
            noise_value = noise_t1.flatten()[i]
            new_value = t2[i]
            if orig_value != new_value:
                print(f"Fails! orig_value={orig_value}, new_value={new_value}, image_value={image_value}, noise_value={noise_value}")
                return False

            print(f"Passed! orig_value={orig_value}, new_value={new_value}, image_value={image_value}, noise_value={noise_value}")
        return True

    #test_clip = [(0, 36), (1, 4), (2, 4), (5, 2)]
    #for tgt_val, noise in test_clip:
    #    print("testing noise = ", noise)
    #    res = test_one(codec, np.full(512 * 3, tgt_val), noise)
    #    if res:
    #        print("Pass!")
    #    else:
    #        print("Fail!")

        

from utils.entropy_coder import MPEGCoder

if __name__ == "__main__":
    #test_ffmpeg()
    #image = Image.open("../debug/pngs/0.png")
    #arr = np.array(image)
    #coder = MPEGCoder()
    #coder.set_qp(0)
    #coder.fit_code_size(arr.size)

    #for i in range(1, 5):
    #    image = Image.open("../debug/pngs/{}.png".format(i))
    #    arr = np.array(image)
    #    bs, vmin, vmax = coder.entropy_encode(arr)
    #    dec_arr = coder.entropy_decode(bs, vmin, vmax).astype("uint8")
    #    print(dec_arr-arr.flatten())
    #    print(len(bs), vmin, vmax)

    #b = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 5, 10])
    #v = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    #print(np.digitize(v, b, True))

    ##list1 = [0] * 10 + [1] * 8 + [2] * 6 + [3] * 4 + [4] * 4 + [5] * 2 + list(range(6, 92)) + [92, 94, 98, 102, 108, 116 ,126]
    ##list1a = [0] * 14 + [1] * 6 + [2] * 6 + [3] * 4 + [4] * 3 + [5] * 2 + list(range(6, 92)) + [92, 94, 98, 102, 108, 116 ,126]
    #list1 = \
    #    [0] * 16 + [1] * 8 + [2] * 8 + [3] * 8 + [4] * 8 + [5] * 4 + [6] * 4 + [7] * 4 + [8] * 4 + [9] * 2 + [10] * 2 + \
    #    list(range(11, 127, 2)) #+ [92, 94, 98, 102, 108, 116 ,126]
    #list1a = \
    #    [0] * 20 + [1] * 8 + [2] * 8 + [3] * 8 + [4] * 6 + [5] * 4 + [6] * 4 + [7] * 4 + [8] * 3 + [9] * 2 + [10] * 2 + \
    #    list(range(11, 127, 2)) #+ [92, 94, 98, 102, 108, 116 ,126]
    #print(len(list1))
    #list2 = np.arange(0, 128)
    #reverse_list1 = np.digitize(list2, list1, True)[1:]
    #print(reverse_list1)

    #test = np.sort(np.random.randint(0, 10, 100))
    #temp = np.digitize(test, list1a, True) + np.random.randint(-5, 5, len(test))
    #rec = np.digitize(temp, reverse_list1, False)
    #print(test)
    #print(temp)
    #print(rec)

    #print(np.unique(test, return_counts=True))
    #print(np.unique(rec, return_counts=True))

    debug_codec()
