"""dlt -- experiment code for the diversity-label trade-off study.

Everything in this package runs on a single GPU. Datasets are kept on the GPU
as uint8 tensors and augmented there in batches, so one training step is
limited by the network and not by data loading.
"""

__version__ = "1.0.0"
