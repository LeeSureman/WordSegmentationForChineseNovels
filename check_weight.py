import pickle
import argparse
from base_tagger import *
parser = argparse.ArgumentParser()
parser.add_argument('--weight_path',default='2_3/trained_weight_18.pkl')
args = parser.parse_args()

weight = pickle.load(open(args.weight_path,'rb'))
print('load successfully')

for name,value in weight.items():
    if name[0] == 'e' and int(name[1])>3:
        print(name,value.now)
