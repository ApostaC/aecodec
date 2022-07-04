import torch
import numpy as np
from sessionconfig import *
from torchvision.transforms.functional import *

class Flow:
    def __init__(self, packets):
        self.packets = packets

    def add_packet(self, packet):
        self.packets.append(packet)

class Packet:
    def __init__(self, data, dropped = False):
        self.data = data
        self.dropped = dropped

    def set_dropped(self, dropped):
        self.dropped = dropped

    def get_dropped(self):
        return self.dropped

    def get_data(self):
        return self.data

    def get_code_post_drop(self):
        if self.dropped:
            return torch.zeros(torch.numel(self.data)).to(self.data.device)
        else:
            return self.data



class Packetizer:

    def packetize(self, packetizer_type, eframe, session_config):
        code = eframe.code
        packet_size = session_config.get_packet_size()
        if packetizer_type == PACKETIER_LINEAR:
            return self.packetize_linear(code, packet_size), eframe
        if packetizer_type == PACKETIER_INTERLEAVED:
            return self.packetize_interleave(code, packet_size), eframe
        else:
            return None, None

    def depacketize(self, packetizer_type, flow, session_config, original_code_size):
        packet_size = session_config.get_packet_size()
        if packetizer_type == PACKETIER_LINEAR:
            return self.depacketize_linear(flow, packet_size, original_code_size)
        if packetizer_type == PACKETIER_INTERLEAVED:
            return self.depacketize_interleave(flow, packet_size, original_code_size)
        else:
            return None

    def packetize_linear(self, code, packet_size):
        leng = torch.numel(code)
        if leng % packet_size == 0:
            num_packets = int(leng / packet_size)
            code_pad = code
        else:
            num_packets = int(leng / packet_size) + 1
            code_pad = torch.zeros(num_packets * packet_size)
            code_pad[:leng] = code
        matrix = code_pad.view(num_packets, packet_size)
        packets = []
        for i in range(num_packets):
            packet = Packet(matrix[i])
            packets.append(packet)
        return Flow(packets)

    def depacketize_linear(self, flow, packet_size, original_code_size):
        code_pad = torch.empty(0).to(flow.packets[0].get_data().device)
        for i in range(len(flow.packets)):
            code_pad = torch.cat((code_pad,
                            flow.packets[i].get_code_post_drop()), 0)
        code = code_pad[:original_code_size]
        return code

    def packetize_interleave(self, code, packet_size):
        leng = torch.numel(code)
        if leng % packet_size == 0:
            num_packets = int(leng / packet_size)
            code_pad = code
        else:
            num_packets = int(leng / packet_size) + 1
            code_pad = torch.zeros(num_packets * packet_size)
            code_pad[:leng] = code
        matrix = code_pad.view(packet_size, num_packets)
        packets = []
        for i in range(num_packets):
            packet = Packet(matrix[:,i])
            packets.append(packet)
        return Flow(packets)

    def depacketize_interleave(self, flow, packet_size, original_code_size):
        num_packets = len(flow.packets)
        code_pad = torch.empty(0).to(flow.packets[0].get_data().device)
        for i in range(num_packets):
            code_pad = torch.cat((code_pad,
                            flow.packets[i].get_code_post_drop()), 0)
        matrix = code_pad.view(num_packets, packet_size)
        code_pad = torch.empty(0).to(flow.packets[0].get_data().device)
        for i in range(packet_size):
            code_pad = torch.cat((code_pad, matrix[:,i]), 0)
        code = code_pad[:original_code_size]
        return code
