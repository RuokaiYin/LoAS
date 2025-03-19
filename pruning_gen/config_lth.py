import argparse

def get_args():
    parser = argparse.ArgumentParser("SNN-LTH")
    parser.add_argument('--exp_name', type=str, default='snn_pruning',  help='experiment name')
    parser.add_argument('--data_dir', type=str, default='/gpfs/gibbs/project/panda/shared/', help='path to the dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10, cifar100]')
    parser.add_argument('--seed', default=1234, type=int)

    parser.add_argument('--timestep', type=int, default=4, help='timestep for SNN')
    parser.add_argument('--batch_size', type=int, default= 512, help='batch size')

    parser.add_argument('--arch', type=str, default='resnet19', help='[resnet19]')
    parser.add_argument('--optimizer', type=str, default='sgd', help='[sgd, adam]')
    parser.add_argument('--scheduler', type=str, default='cosine', help='[step, cosine]')
    parser.add_argument('--learning_rate', type=float, default=0.3, help='learnng rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')

    # for lth
    parser.add_argument("--prune_percent", default=25, type=float, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=15, type=int, help="Pruning iterations count")
    parser.add_argument('--round', type=int, default=4, help='for mean and std === 1 to 5')
    parser.add_argument("--pruning_scope", default="global", type=str)
    parser.add_argument("--epoch", default=150, type=int)
    parser.add_argument('--valid_freq', type=int, default=10, help='test for SNN')
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--rewinding_epoch", default=20, type=int)

    args = parser.parse_args()
    print(args)

    return args
