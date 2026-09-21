## vramxray

A memory debugger for PyTorch on nvidia gpus. It explains why a cuda job ran out of memory, names the memory torch cannot see, and points at the line to change.

### Usage/Install

```bash
pip install vramxray            # pure Python
pip install "vramxray[native]"  # adds the C++ core
```

```python
import vramxray

vramxray.watch()
```

That is the whole setup. 

example oom traceback from vramxray

```
vramxray: OOM on cuda:0, requesting 256.0 MiB
  device (NVML)           4.91 GiB used of 15.92 GiB
    torch reserved          2.95 GiB   allocated 2.83 GiB · free in segments 121.9 MiB
    libnccl.so.2          388.0 MiB
    libcublas.so.12         16.3 MiB
    kernel images           64.0 MiB   libtorch_cuda.so 64.0 MiB
    CUDA context           275.1 MiB   measured across torch.cuda.init()
    peak so far             3.10 GiB   reserved at t=41s, 154.0 MiB above now
    allocator stalls           144 ms   over 96 driver calls, mostly libc10_cuda.so

  why 120.8 MiB free inside torch could not serve 256.0 MiB:
    verdict: exhaustion. Even after returning every cached block to the driver, 85.2 MiB would
      still be missing.
    live memory (2.83 GiB) by call site:
        2.00 GiB    70%    105 blocks  train.py:30 forward from train.py:38 <module>
       256.0 MiB     9%      2 blocks  train.py:39 <module>
       214.4 MiB     7%    152 blocks  train.py:41 <module>

  try:
    70% of live memory (2.00 GiB) comes from train.py:30 forward. That is the place to shrink:
      smaller batch, activation checkpointing, or offloading
```

## API

```python
import vramxray
from vramxray import streams

vramxray.watch(stacks="python")      # a file and line for every call site
vramxray.report()                    # the same report any time, no OOM needed
vramxray.history().to_polars()       # reserved, live and largest free block over time
vramxray.history().peak(0)           # high water mark, and when it happened
vramxray.timeline().to_polars()      # every allocator and driver event as a table
vramxray.stalls()                    # seconds inside driver allocation calls, per library
vramxray.analyze("dump.pickle")      # explain a snapshot offline, no GPU needed
vramxray.release_cupti()             # hand CUPTI back before a torch.profiler session

streams.start(channels=True)         # blocking streams, broken graph captures, false deps
streams.report()
```

```bash
vramxray analyze dump.pickle --request 1.5G
vramxray run train.py
vramxray merge vramxray-oom-rank.json
```
