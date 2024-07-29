# LoAS

The exploration of the design space of spMspM acceleration for dual sparse SNNs.

This repo intends to provide the source codes in PyTorch for fine-tuning and profiling the SNN models.

1a. Profiling the SNN models to examine the original ratio of silent neurons.\
   python3 model_profile.py -profile --n_mask 0

1b. Profiling the SNN models to examine the ratio of silent neurons by masking out all neurons that only spike for 1 time.\
   python3 model_profile.py -profile --n_mask 1

\
2. Finetuning the SNN models to recover the accuracy from masking out the neurons that only spike for 1 time.\
   python3 fine_tune.py --n_masks 1


Package version:

Python 3.9.7.\
CUDA 11.1.\
PyTorch 2.3.1 py3.9_cuda11.8_cudnn8.7.0_0\
spikingjelly 0.0.0.0.12

More details to come soon.

