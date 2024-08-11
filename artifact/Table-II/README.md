# Artifact results reproduce for Table.II

## How to run?

### Please note that at the end of the script, there will be a ``clear`` command!!! Make sure there is nothing important on your console before you run the script.

Simply type the commands below inside the current directory

``/bin/bash run.sh``

The truncated results that is solely for the artifact purpose will be generated inside the ``tableII_artifact.txt`` file.


## How to check?
The layerwise and the network workload data are all written in the ``tableII_artifact.txt`` file. This is corresponding to the Table.II in the paper, please feel free to check the reproduced results with the one in the paper.

We have tested running the script on a x86_64 machine and provide a reference inside the ``reference.txt``. Please note that, there will be several entries that have very small deviations (~0.1%) due to the rounding.
