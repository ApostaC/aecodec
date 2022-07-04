import numpy as np
from typing import List

"""
=====================
helpers
=====================
"""

def find_suitable_size(data, entropy_coder, hint):
    """
    return the suitable code, encoded size and the origin size
    """
    pkt_sz = Packet.MAX_SIZE
    # try encode the whole
    bs, size = entropy_coder.entropy_encode(data)
    if size < pkt_sz:
        return bs, size, data.size

    # find the suitable size
    mid = hint
    tries = 0
    while tries < 5:
        bs, size = entropy_coder.entropy_encode(data[:mid])
        #print("Hint is: {}, size is {}".format(mid , size))
        if size > pkt_sz or size < 0.9 * pkt_sz:
            mid = int(mid / size * (0.95 * pkt_sz))
        else:
            break
        tries += 1
    while True:
        bs, size = entropy_coder.entropy_encode(data[:mid])
        #print("Hint is: {}, size is {}".format(mid , size))
        if size > pkt_sz:
            mid = int(mid * 0.8)
        else:
            break
    bs, size = entropy_coder.entropy_encode(data[:mid])
    return bs, size, mid

    # find the upper bound
    #upper_bound = hint
    #while True:
    #    bs, size = entropy_coder.entropy_encode(data[:upper_bound])
    #    if size > pkt_sz:
    #        break
    #    else:
    #        upper_bound = int(upper_bound * 1.5)

    ## find the lower bound
    #lower_bound = hint
    #while True:
    #    bs, size = entropy_coder.entropy_encode(data[:lower_bound])
    #    if size < pkt_sz:
    #        break
    #    else:
    #        lower_bound = int(lower_bound / 1.5)

    ## find the suitable size
    #while True:
    #    mid = (upper_bound + lower_bound) // 2
    #    bs, size = entropy_coder.entropy_encode(data[:mid])
    #    if size > pkt_sz:
    #        upper_bound = mid
    #    elif size < 0.95 * pkt_sz:
    #        lower_bound = mid
    #    else:
    #        break
    #bs, size = entropy_coder.entropy_encode(data[:mid])
    #return bs, size, mid

"""
=====================
main functions
=====================
"""
class Packet:
    MAX_SIZE = 1400

    def __init__(self):
        self.frame_id = 0
        self.frame_type = 0 # 1 is I-frame, 2 is P-frame
        self.frac_id = -1
        self.frac_cnt = 0
        self.origin_size = 0
        self.payload = np.zeros(0)
        self.is_lost = 0

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Packet(frame_id={self.frame_id}, frame_type={self.frame_type}, frac_id={self.frac_id}, frac_cnt={self.frac_cnt}, orig_size={self.origin_size}, is_lost={self.is_lost}, payload_size={len(self.payload)}\n"

def Packetize(data: np.ndarray, entropy_coder, frame_id, frame_type):
    """
    Parameters:
        data: the input data in 1d numpy array
        entropy_coder: can encode the data
        frame_id: the id of the frame
        frame_type: 1 or 2 (I or P)
    Returns:
        a list of Packet object
    """
    pkt_sz = Packet.MAX_SIZE
    data = data.flatten()

    stream, size = entropy_coder.entropy_encode(data)
    compress_ratio = size / data.size 

    # initialize the hint using compress_ratio
    hint = int(pkt_sz / compress_ratio)
    offset = 0

    packets = []
    pkt_id = 0
    while offset < data.size:
        remaining_size = data.size - offset
        bs, size, mid = find_suitable_size(data[offset:], entropy_coder, hint)
        offset += mid
        hint = mid

        # generate the packet
        pkt = Packet()
        pkt.frame_id = frame_id
        pkt.frame_type = frame_type
        pkt.origin_size = mid
        pkt.frac_id = pkt_id
        pkt_id += 1
        pkt.payload = bs
        packets.append(pkt)

    for pkt in packets:
        pkt.frac_cnt = len(packets)

    return packets

def Depacketize(packet_group: List[Packet], entropy_coder):
    # do the parameter check!
    assert packet_group, "No packets in the packet group!"
    frac_cnt = packet_group[0].frac_cnt
    assert len(packet_group) == frac_cnt, f"Not enough packets in group, expect {frac_cnt} but got {len(packet_group)}"
    
    # recover the original data using the entropy coder
    # if packet is lost, then the values are all zero!
    decoded_pkts = []
    for pkt in packet_group:
        if pkt.is_lost:
            decoded_pkts.append(np.zeros(pkt.origin_size))
        else:
            code = entropy_coder.entropy_decode(pkt.payload)
            if len(code) != pkt.origin_size:
                print("Warning: decoded code size is not equal to the original size!")
            decoded_pkts.append(code)
    origin_stream = np.concatenate(decoded_pkts, axis = None)
    return origin_stream


