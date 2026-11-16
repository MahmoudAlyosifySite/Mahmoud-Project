"""dlt -- experiment code for the diversity-label trade-off study.

Everything in this package runs on a single GPU. Datasets are kept on the GPU
as uint8 tensors and augmented there in batches, so one training step is
limited by the network and not by data loading.
"""

import os as _os

# fewer out-of-memory failures from fragmentation when two runs share the GPU (must be set before CUDA starts)
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "1.0.0"
