# Artifact results reproduce for Table.II

## How to run?

### Please note that at the end of the script, there will be a ``clear`` command!!! Make sure there is nothing important on your console before you run the script.

Simply type the commands below inside the current directory

``/bin/bash run.sh``

The truncated results that is solely for the artifact purpose will be generated inside the ``tableII_artifact.txt`` file.


## How to check?
The layerwise and the network workload data are all written in the ``tableII_artifact.txt`` file. This is corresponding to the Table.II in the paper, please feel free to check the reproduced results with the one in the paper.

We have tested running the script on a x86_64 machine and provide a reference inside the ``reference.txt``. Please note that, there will be several entries that have very small deviations (~0.1%) due to the rounding.

## How to interpret the generated results?
Inside the ``tableII_artifact.txt``, there will be in total 6 blocks of results, representing the results of two modes (without and with fine-tuning) for each of the 3 different SNN models.

Each block will have in total of 8 lines:

1. Title of the model name and the mode type. Normal means the model without fine-tuning and strong means the model with fine-tuning.
2. Layer number for the layerwise results. Please note that, the layer number here in the text file will always be one layer less compared to the one in Table.II from the original paper. The reason is that we are not considering the first layer (encoding layer) in our profiling.
3. This line corresponds to the 4th column (AvSpA origin) for layerwise workloads(A-L4, V-L8, and R-L19) in Table.II from the original paper. For each SNN model, this number should be identical for both modes.
4. This line corresponds to the 5th column for layerwise workloads in Table.II from the original paper. This line will be different for two modes. For the mode of normal, this line corresponds to the results outside the bracket. For example, in the 4th line of the generated text file, it shows the ratio of silent neurons is 63.1922%, this will correspond to the number that is outside the bracket in the 5th column for A-L4 in Table.II, which is 63.2. For the mode of strong (2nd blocks for each model), its 4th line will correspond to the number inside the bracket of 5th column for layerwise workloads. For example, the 69.6719% in the 13th line of the generated text file correspond to the 69.7(inside the bracket) in the 5th column for A-L4 in Table.II.
5. This line corresponds to the 6th column (AvSpB) for layerwise worklaods in Table.II from the original paper. Similarly, for each SNN model, this number should be identical for both modes.
6. This line corresponds to the 4th line in Table.II for the entire SNN model (AlexNet(A), VGG16(V), and ResNet19(R)). This number is same for both modes.
7. Similar to line 4, this number is different for both modes, but this time, corresponds to the entire SNN models in Table.II. For example, 71.2559% in line 7 corresponds to 71.3(outside the bracket) in column 5 for AlexNet(A).
8. This line correspond to the weight sparsity for the entire SNN model (6th column in Table.II). This number is also same between two modes. 
