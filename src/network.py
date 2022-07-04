import torch
import numpy as np
from sessionconfig import *
from torchvision.transforms.functional import *


class Link:
    def __init__(self, loss):
        self.loss = loss
        self.packet_size = 1 # placeholder

    def send(self, flow, session_config):
        flow.set_packet_size(self.packet_size)
        flow.set_loss_rate(self.loss)
        return flow

class Flow:
    def __init__(self, size_in_bytes):
        self.size_in_bytes = size_in_bytes
        self.loss_rate = 0
        self.packet_size = 0

    def get_size_in_bytes(self):
        return self.size_in_bytes

    def set_loss_rate(self, loss_rate):
        self.loss_rate = loss_rate

    def get_loss_rate(self):
        return self.loss_rate

    def set_packet_size(self, packet_size):
        self.packet_size = packet_size

    def get_packet_size(self):
        return self.packet_size

class Packetizer:

    def to_flow(self, eframe, model, session_config):
        # potentially we can transform the code field of eframe
        size_in_bytes = model.get_size_in_bytes(eframe, session_config)
        w, h = session_config.get_frame_size()
        bpp = size_in_bytes * 8 / (w * h)
        return Flow(size_in_bytes), bpp, eframe

    def from_flow(self, flow, eframe, session_config):
        if eframe.code == None:
            return eframe
        blocksize = 1
        loss_ratio = flow.get_loss_rate()
        leng = torch.numel(eframe.code)
        nblocks = (leng - 1) // blocksize + 1
        rnd = torch.rand(nblocks).to(eframe.code.device)
        rnd = (rnd > loss_ratio).long()
        rnd = rnd.repeat_interleave(blocksize)
        rnd = rnd[:leng].reshape(eframe.code.shape)
        eframe.code = eframe.code * rnd
        return eframe
