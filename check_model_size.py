import torch
from models.archs.GFCS_X_mod1 import GFCSNetwork

TOTAL_SIZE = 0

def get_module_size(module):
    # Only count parameters and buffers defined directly in this module.
    param_size = sum(p.nelement() * p.element_size() for p in module.parameters(recurse=False))
    buffer_size = sum(b.nelement() * b.element_size() for b in module.buffers(recurse=False))
    total_size = (param_size + buffer_size) / (1024 ** 2)   # in MB
    return total_size

def print_component_sizes(model):
    global TOTAL_SIZE
    print("Component memory usage breakdown (parameters+buffers in MB):")
    for name, module in model.named_modules():
        size = get_module_size(module)
        if size > 0:
            print(f"{name if name else 'ROOT'}: {size:.4f} MB")
        TOTAL_SIZE += size
    print(f"Total: {TOTAL_SIZE:.4f} MB")

def get_model_size(model):
    param_size = 0
    param_count = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_count += param.nelement()
    buffer_size = 0
    buffer_count = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_count += buffer.nelement()
    
    total_size = (param_size + buffer_size) / 1024**2  # Convert to MB
    return total_size, param_count, buffer_count

# New: Hook to capture the size of module outputs (approximate activation memory)
def activation_mem_hook(module, input, output):
    # Compute memory usage for the output tensors (assumes tensor or list/tuple of tensors)
    if isinstance(output, torch.Tensor):
        mem = output.nelement() * output.element_size() / (1024**2)
    elif isinstance(output, (list, tuple)):
        mem = sum(o.nelement() * o.element_size() for o in output if isinstance(o, torch.Tensor)) / (1024**2)
    else:
        mem = 0
    module.activation_mem = mem

def measure_activation_memory(model, dummy_input):
    hooks = []
    activation_memory = {}
    # Register hook on every module
    for name, module in model.named_modules():
        hook = module.register_forward_hook(lambda m, i, o, name=name: activation_mem_hook(m, i, o))
        hooks.append(hook)
    model(dummy_input)
    for name, module in model.named_modules():
        if hasattr(module, 'activation_mem'):
            activation_memory[name if name else 'ROOT'] = module.activation_mem
    for hook in hooks:
        hook.remove()
    return activation_memory

if __name__ == "__main__":
    model = GFCSNetwork(inp_channels=3, out_channels=3, dim=48)
    model.half()  # Ensure model parameters use float16
    total_size, param_count, buffer_count = get_model_size(model)
    print("================ Model memory usage breakdown (parameters+buffers in MB) ================")
    print(f"Total Model Size: {total_size:.2f} MB")
    print(f"Total Parameters: {param_count}")
    print(f"Total Buffers: {buffer_count}\n")
    
    print_component_sizes(model)
    
    # Measure intermediate activation memory using a dummy input.
    # Adjust the dummy input shape as needed—here we assume (B, 3, H, W)
    dummy_input = torch.randn(1, 3, 256, 256, device='cuda', dtype=torch.half)
    model.to('cuda')
    model.eval()
    activation_memory = measure_activation_memory(model, dummy_input)
    
    print("\n================ Activation memory usage breakdown (output tensor sizes in MB) ================")
    print("\nApproximate Activation Memory per Component (output tensor sizes in MB):")
    total = 0
    for name, mem in activation_memory.items():
        total += mem
        print(f"{name}: {mem:.4f} MB")
    print(f"Total: {total:.4f} MB")
