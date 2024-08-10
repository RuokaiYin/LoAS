python3 ../../fine_tune.py --n_mask 1 --arch vgg16 --learning_rate 0.02
python3 ../../fine_tune.py --n_mask 1 --arch resnet19 --learning_rate 0.05
python3 plot.py
rm FT_artifact.txt