import argparse


def get_args():
    parser = argparse.ArgumentParser("LoAS-Profile-SNN")
    
    #! Manually set your own data_dir
    parser.add_argument('--data_dir', type=str, default='./dataset', help='path to the dataset')
    
    #! Set to different values if required.
    parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10]')
    parser.add_argument('--arch', type=str, default='resnet19', help='[vgg16, resnet19, alexnet]')
    parser.add_argument('--timestep', type=int, default=4, help='timestep for SNN')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')

    #! Turn on for detailed profiling data on the silent neurons.
    parser.add_argument('-profile',action='store_true')

    #! Default 0 means no maunally silencing the neuron.
    parser.add_argument('--n_mask', type=int, default=0, help='threshold of # of spikes to silent a neuron')

    args = parser.parse_args()
    return args
