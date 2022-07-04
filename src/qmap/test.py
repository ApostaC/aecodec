import torch
import numpy as np
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from qmap_model import QmapModel
from PIL import Image, ImageFile, ImageFilter


config = {
        "N": 192, 
        "M": 96,
        "sft_ks": 3,
        "name": "models4x",
        "path": "/datamirror/yihua98/projects/autoencoder_testbed/models/qmap/prev_8x.pt"
    }

if __name__ == "__main__":
    qmap_encoder = QmapModel(config)
    image = Image.open("/datamirror/yihua98/projects/autoencoder_testbed/data/pole_small/0.png").convert("RGB")
    image = to_tensor(image)

    code, shapex, shapey = qmap_encoder.encode(image)
    out = qmap_encoder.decode(code, shapex, shapey)
    #print(out)
    #out = qmap_encoder.decode(code)
    #print(code["strings"][0].shape, code["strings"][1].shape)
    #print(out["x_hat"].shape)

    np_code = code.cpu().detach().numpy()
    np.save("../../debug/qmap_code.npz", np_code.flatten())
    
    save_image(out, "../../debug/test_output.png")
