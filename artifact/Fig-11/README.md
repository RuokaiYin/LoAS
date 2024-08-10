# Artifact results reproduce for Fig.11

## How to run?
 Simply type the commands below inside the current directory

``/bin/bash run.sh``

The script will automatically run the fine-tuning for VGG16 and ResNet19 on CIFAR10 dataset. The result will be a figure named ft_accuracy.pdf, which is esentially the Figure 11 in the original paper.

Please kindly note that, the generated figure will not exactly match the number of the figure in the paper. This is due to the randomness in the model and fine-tuning. 

However, it is very clear that the trend retains the same as in the original figure, and clearly match what we mentioned in the paper "We find that with a very small number of fine-tuning
(<5 epochs), the accuracy can be fully recovered..."

We provide some ``reference-{device}.pdf`` figures for references. These are the fresh figures I re-generated from the script on different devices before the artifact submission.

## Hyperparameter takeaway
The shared training hyperparameter can be found in the ``config_ft.py`` (outside the artifact directory).

The specific learning rate for VGG16 is 0.02, for ResNet19 is 0.05.

We only provide a basic hyperparameter combination here, different combinations will have different impact on the result.

## Machine testing

The script has been tested on x86_64 machines below:

#### NVIDIA RTX 2080 Ti GPU + Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz
Roughly takes 16.5 minutes to get the result.

#### NVIDIA V100 GPU + Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz
Roughly takes 10 minutes to get the result.
