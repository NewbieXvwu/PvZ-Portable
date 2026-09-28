"""Probe the Windows-side Python/torch/CUDA stack of the desktop machine."""
import platform
import sys

print("python      :", sys.version.replace("\n", " "))
print("executable  :", sys.executable)
print("platform    :", platform.platform())
print("machine     :", platform.machine())
print()

try:
    import torch
except Exception as exc:  # pragma: no cover - diagnostic path
    print("torch       : IMPORT FAILED ->", type(exc).__name__, exc)
    raise SystemExit(0)

print("torch       :", torch.__version__)
print("cuda build  :", torch.version.cuda)
print("cudnn       :", torch.backends.cudnn.version())
print("cuda avail  :", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("arch list   :", torch.cuda.get_arch_list())
print("threads     :", torch.get_num_threads(), "interop", torch.get_num_interop_threads())
try:
    import numpy
    print("numpy       :", numpy.__version__)
except Exception as exc:
    print("numpy       : IMPORT FAILED ->", type(exc).__name__, exc)

if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"gpu[{index}]     : {props.name}  cc={props.major}.{props.minor}  "
              f"mem={props.total_memory / 2**30:.1f} GiB  sm_count={props.multi_processor_count}")
    # Does a real kernel actually run on sm_120?
    try:
        x = torch.randn(512, 512, device="cuda")
        y = (x @ x).sum().item()
        print("cuda matmul : OK ->", y)
    except Exception as exc:
        print("cuda matmul : FAILED ->", type(exc).__name__, exc)
