import torch
from torch import nn
from thop import profile
from pthflops import count_ops
from fvcore.nn import FlopCountAnalysis
import numpy as np
from flopth import flopth

def get_macs_params(model, input_ = torch.randn(1, 3, 224, 224)):
    macs, params = profile(model, inputs=(input_, ))
    return macs, params

# def get_flops(model, input_ = ((3, 224, 224),)):
#     model.eval()
#     flops, params = flopth(model, input_)
#     flops = float(flops[:-1])
#     return flops*(1000000)
    
    
def get_flops(model, input_ = torch.randn(1, 3, 224, 224)):
    model.eval()
    flops = FlopCountAnalysis(model, input_)
    return flops.total()

def measure_latency(model, device_type):
    
    model.eval()
    if device_type == 'gpu':
        device = torch.device("cuda")
        
    elif device_type == 'cpu':
        device = torch.device("cpu")

    else:
        print("Device not found. Exit")
        exit()
        
    model.to(device)
    dummy_input = torch.randn(1,3,224,224, dtype=torch.float).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    
    return mean_syn




if __name__ == '__main__':
    from torchvision.models import resnet101
    model = resnet101()

    cpu_latency_mean = measure_latency(model, device_type = 'cpu')
    gpu_latency_mean = measure_latency(model, device_type = 'gpu')
    
    






