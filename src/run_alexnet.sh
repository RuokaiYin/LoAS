#!/bin/sh

python3 snn-sparten-result.py --arch alexnet | tee snn_sparten_alexnet_result.txt
python3 snn-gospa-result.py --arch alexnet | tee snn_gospa_alexnet_result.txt
python3 snn-loas-mntk-result.py --arch alexnet --loas strong | tee snn_loas-mntk-strong_alexnet_result.txt
python3 snn-loas-nmtk-result.py --arch alexnet --loas strong | tee snn_loas-nmtk-strong_alexnet_result.txt
python3 snn-loas-mntk-result.py --arch alexnet --loas normal | tee snn_loas-mntk-normal_alexnet_result.txt
python3 snn-loas-nmtk-result.py --arch alexnet --loas normal | tee snn_loas-nmtk-normal_alexnet_result.txt
