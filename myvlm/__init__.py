import pyrallis
import torch

def decode_dtype(dtype: str) -> torch.dtype:
    if dtype == 'float16':
        return torch.float16
    elif dtype == 'bfloat16':
        return torch.bfloat16
    else:
        return torch.float32

pyrallis.decode.register(torch.dtype, lambda x: decode_dtype(x))
pyrallis.encode.register(torch.dtype, lambda x: x.__str__())
