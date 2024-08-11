import argparse

def get_args():
    parser = argparse.ArgumentParser("SNN-FineTune")
    parser.add_argument('--exp_name', type=str, default='snn_pruning', help='experiment name')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='path to the dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10]')
    parser.add_argument('--seed', default=428, type=int)

    parser.add_argument('--timestep', type=int, default=4, help='timestep for SNN')
    parser.add_argument('--batch_size', type=int, default= 128, help='batch size')

    parser.add_argument('--arch', type=str, default='resnet19', help='[vgg16, resnet19, alexnet]')
    parser.add_argument('--optimizer', type=str, default='sgd', help='[sgd, adam]')
    parser.add_argument('--scheduler', type=str, default='cosine', help='[step, cosine]')
    parser.add_argument('--learning_rate', type=float, default=2e-2, help='learnng rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument("--epoch", default=50, type=int)


    #! Default 0 means no maunally silencing the neuron.
    parser.add_argument('--n_mask', type=int, default=0, help='threshold of # of spikes to silent a neuron')

    #! For artifact purpose
    parser.add_argument('--artifact', action='store_true', help='Usin this key to turn on the artifact mode.')

    #! For ResNet19
    #* Batch size = 128, sgd, cosine, lr=5e-2, momentum=0.9, weight-decay=5e-4
    args = parser.parse_args()
    # print(args)

    return args
