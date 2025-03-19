# LoAS -- LTH-based Pruning framework

### This sub-directory is created to provide some basic supports for the users to generate their own LTH-based prunned SNNs.

### Environments
If you have already installed the environments as we provided in the main directory, you should be good to go.

### How to use?
Simply run the command ``python3 prune_train.py`` to begin your exploration. The codes will automatically create a folder called ``lth`` and store the prunned models and masks inside. All the hyperparameters can be found in the ``config_lth.py``.

For testing the codes, I threw some random hyperparameters and ran the codes for 12 pruning iterations (each with 150 epochs). The reference accuracy and sparsity results can be found in the ``result.txt``. I ran the code with a single V100 GPU.

Currently, I only provide a basic ResNet19 architecture as an example. Later I can provide the fully-temporal-parallel based architectures as in the main paper. But the pruning part is almost the same. You are also encouraged to bring your own architectures and use our codes to prune them.

### Acknowledgement
The codes is adopted from Youngeun's 2022 ECCV paper ``Exploring Lottery Ticket Hypothesis in Spiking Neural Networks``.

You can also find an optimized version which ensures the prunned workload to be 100% balanced in another work of mine: 
``https://github.com/RuokaiYin/u-Ticket``


### Citing

If you find the above code is useful for your research, please use the following bibtex to cite us,
```bibtex
@inproceedings{yin2024loas,
  title={LoAS: Fully Temporal-Parallel Dataflow for Dual-Sparse Spiking Neural Networks},
  author={Yin, Ruokai and Kim, Youngeun and Wu, Di and Panda, Priyadarshini},
  booktitle={2024 57th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  pages={1107--1121},
  year={2024},
  organization={IEEE}
}