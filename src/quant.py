import torch
from functools import partial

class FQuantizer:
    def __init__(self, f, g, input_range):
        """
        Input:
            f: the quant function, takes a tensor as the input and returns the quanted tensor
            g: the reverse of f
        """
        self.f = f
        self.g = g
        self.input_range = input_range

    def quantize(self, vec):
        """
        Input:
            vec: torch tensor
        Output:
            bs: quantized tensor, bs[i] = round(f[vec[i]])
        """
        vec.clamp_(-self.input_range, self.input_range)
        return self.f(vec)

    def dequantize(self, bs):
        """
        Input:
            bs: the quantized tensor
        Output:
            vec: the tensor after quantization
        """
        return self.g(bs)

    def get_output_range(self):
        """
        Returns:
            The range of values after quantization
        """
        vals = torch.arange(-self.input_range, self.input_range + 1)
        vec = self.f(vals)
        low, high = vec.min(), vec.max()
        return torch.floor(low).item(), torch.ceil(high).item()

    def get_index(self, real_val, low_range, high_range):
        """
        Input:
            real_val: the real value
            low_range: inclusive
            high_range: inclusive
        Return:
            the normalized index 
        """
        return real_val - low_range

    def process_distribution(self, G: torch.tensor):
        """
        Input:
            G: the CDF distribution, G[i] = P(x <= i - input_range), i in [-input_range, input_range]
        Output:
            G': a new CDF distribution, G'[i] = P(y <= i - output_low), i in [output_low, output_high], y = f(x)
        """
        vals = torch.arange(-self.input_range, self.input_range + 1)
        vec = self.f(vals)
        low, high = torch.floor(vec.min()).item(), torch.ceil(vec.max()).item()
        low = int(low)
        high = int(high)
        
        Gp = torch.zeros(int(high - low + 2))
        for i in range(low, high + 2):
            a = self.get_index(i, low, high)
            y = a + low
            x = self.g(torch.tensor(y).to(torch.float)).to(torch.int).clamp(-self.input_range, self.input_range).item()
            x = int(round(x))
            b = self.get_index(x, -self.input_range, self.input_range)
            Gp[a] = G[b]
        Gp[-1] = 1
        return Gp
    

ALPHA = 48
BETA = 0.044
def quantFunc(v : torch.tensor):
    global ALPHA, BETA
    v = torch.exp((-1)*BETA*v)
    v = ALPHA/(1 + v)
    return torch.round(v)

def dequantFunc(v : torch.tensor):
    global ALPHA, BETA
    v = -torch.log(1 / (v / ALPHA) - 1 + 1e-6) / BETA
    #return torch.round(v)
    return v

def dequantFunc_gen(v : torch.tensor, mxrange):
    labels = torch.arange(-mxrange, mxrange+1)
    vec = quantFunc(labels)
    ret = torch.bucketize(v, vec, right=True)
    return ret


def getDefaultQuantizer(mxrange):
    #tmp = partial(dequantFunc_gen, mxrange=mxrange)
    return FQuantizer(f=quantFunc, g=dequantFunc, input_range=mxrange)
    
if __name__ == "__main__":
    def f(v: torch.tensor):
        return torch.round(v / 5)

    def g(v: torch.tensor):
        return torch.round(v * 5)

    quantizer = FQuantizer(f=f, g=g, input_range=20)
    print(quantizer.get_output_range()) # -4, 4
    print(quantizer.get_index(1, -5, 8)) # 6

    input_dis = torch.full((41, ), 1 / 41)
    input_dis = torch.cumsum(input_dis, dim = 0)
    print(input_dis)

    output_dis = quantizer.process_distribution(input_dis)
    print(output_dis)

    quantizer = getDefaultQuantizer(150)
    print(quantizer.get_output_range())

    v1 = quantizer.quantize(torch.arange(-150, 150))
    v2 = quantizer.dequantize(v1)
    print(v1)
    _, cnts = v1.unique(return_counts=True)
    print(torch.cumsum(cnts, dim=0))
    print(v2)

    input_dis = torch.full((301, ), 1 / 301)
    input_dis = torch.cumsum(input_dis, dim = 0)
    output_dis = quantizer.process_distribution(input_dis)

    print(input_dis)
    print(output_dis)
