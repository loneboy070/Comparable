import argparse 

parser = argparse.ArgumentParser(description='argparse manual')


parser.add_argument('--N_TRAIN', required=False, default=500, help='fuck')
parser.add_argument('--N_TEST', required=False, default=300, help='fuck')
parser.add_argument('--FEATURE_DIM', required=False, default=5, help='fuck')


parser.add_argument('--ENN', required=False, default = True)

parser.add_argument('--Node_Sizes', required=False, default=[3, 3, 1])

# parser.add_argument('--target', required=False, help='fuck')