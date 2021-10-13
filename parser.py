import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Amazon-Books',help='which dataset to use (music, book, ml-10m)')
    parser.add_argument('--n_epoch', type=int, default=500, help='the number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048,help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--dim', type=int, default=64, help='dimension of embeddings')
    parser.add_argument('--n_hops', type=int, default=3, help='depth of layer')
    parser.add_argument('--l2', type=float, default=1e-5, help='weight of the l2 regularization term')

    parser.add_argument('--agg',type=str, default='attention', help='the type of aggregator (GNN, Path, attention)')

    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")                     
    parser.add_argument('--node_dropout_rate', type=float, default=0.5, help='node dropout rate')    
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=3, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[10, 20, 40, 60, 80]', help='top-K list')
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="batch items")
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')


    parser.add_argument('--ls', type=float, default=1e-3, help='weight of the seperation loss')
    parser.add_argument('--lu', type=float, default=1e-1, help='weight of the user specific loss')
    parser.add_argument("--save", type=bool, default=True, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./save/", help="output directory for model")

    args = parser.parse_args()
    return args