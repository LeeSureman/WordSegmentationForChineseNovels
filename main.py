import argparse


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--train', required=True, help='training set')
    parser.add_argument('--test', default=None, help='test set')
    parser.add_argument('--record', default=None, help='file output to record measure')
    parser.add_argument('--weight', default=None, help='weight output to')
    parser.add_argument('--start', default=0, type=int, help='set the round')
    parser.add_argument('--epoch', default=1, type=int, help='as the name')
    parser.add_argument('--init_weight',required=False,default=None,help='initial feature weight')
    args = parser.parse_args()