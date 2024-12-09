import torch
from torch import nn


def estimate_memory_training(model, sample_input = torch.randn(1,3,224,224), 
                             optimizer_type=torch.optim.Adam, batch_size=256, use_amp=False, device=0):
    
    model.cpu()
    optimizer = optimizer_type(model.parameters(), lr=.001)
    a = torch.cuda.memory_allocated(device)
    model.to(device)
    b = torch.cuda.memory_allocated(device)
    model_memory = b - a
    model_input = sample_input
    output = model(model_input.to(device)).sum()
    c = torch.cuda.memory_allocated(device)
    if use_amp:
        amp_multiplier = .5
    else:
        amp_multiplier = 1
    forward_pass_memory = (c - b)*amp_multiplier
    gradient_memory = model_memory
    if isinstance(optimizer, torch.optim.Adam):
        o = 2
    elif isinstance(optimizer, torch.optim.RMSprop):
        o = 1
    elif isinstance(optimizer, torch.optim.SGD):
        o = 0
    elif isinstance(optimizer, torch.optim.Adagrad):
        o = 1
    else:
        raise ValueError("Unsupported optimizer. Look up how many moments are" +
            "stored by your optimizer and add a case to the optimizer checker.")
    gradient_moment_memory = o*gradient_memory
    total_memory = model_memory + forward_pass_memory + gradient_memory + gradient_moment_memory

    return total_memory


        
def estimate_memory_inference(model, sample_input = torch.randn(1,3,224,224), batch_size=256, use_amp=False, device=0):
    model.cpu()
    a = torch.cuda.memory_allocated(device)
    model.to(device)
    b = torch.cuda.memory_allocated(device)
    model_memory = b - a
    model_input = sample_input
    output = model(model_input.to(device)).sum()
    total_memory = model_memory

    return total_memory    