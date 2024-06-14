#!/bin/sh

python3 snn-sparten-result.py --arch resnet19 | tee snn_sparten_resnet19_result.txt
python3 snn-gospa-result.py --arch resnet19 | tee snn_gospa_resnet19_result.txt
python3 snn-loas-mntk-result.py --arch resnet19 --loas strong | tee snn_loas-mntk-strong_resnet19_result.txt
python3 snn-loas-nmtk-result.py --arch resnet19 --loas strong | tee snn_loas-nmtk-strong_resnet19_result.txt
python3 snn-loas-mntk-result.py --arch resnet19 --loas normal | tee snn_loas-mntk-normal_resnet19_result.txt
python3 snn-loas-nmtk-result.py --arch resnet19 --loas normal | tee snn_loas-nmtk-normal_resnet19_result.txt
