import entropy_coder as EC
import packet as P
import numpy as np

if __name__ == "__main__":
    data = np.load("../../debug/qmap_code.npz.npy")
    print(data)

    # torchac
    coder = EC.TorchacCoder()
    bs, cdf, size, minval = coder.entropy_encode(data)
    out = coder.entropy_decode(bs, cdf, minval)
    print("Torchac:", size, out)

    # zlib
    coder = EC.GzipCoder(np.int8)
    bs, size = coder.entropy_encode(data)
    out = coder.entropy_decode(bs)
    print("gzip: ", size, out)

    # test packetize
    pkts = P.Packetize(data, coder, 1, 1)
    print(pkts)

    bytestream = P.Depacketize(pkts, coder)
    print(len(bytestream))
    for idx, val in enumerate(zip(data, bytestream)):
        orig, dec = val
        if orig != dec:
            print(f"Not equal at index {idx}: original = {orig}, decoded = {dec}")
