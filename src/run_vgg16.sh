#!/bin/sh

python3 snn-sparten-result.py --arch vgg16 | tee snn_sparten_vgg16_result.txt
python3 snn-gospa-result.py --arch vgg16 | tee snn_gospa_vgg16_result.txt
python3 snn-loas-mntk-result.py --arch vgg16 --loas strong | tee snn_loas-mntk-strong_vgg16_result.txt
python3 snn-loas-nmtk-result.py --arch vgg16 --loas strong | tee snn_loas-nmtk-strong_vgg16_result.txt
python3 snn-loas-mntk-result.py --arch vgg16 --loas normal | tee snn_loas-mntk-normal_vgg16_result.txt
python3 snn-loas-nmtk-result.py --arch vgg16 --loas normal | tee snn_loas-nmtk-normal_vgg16_result.txt
