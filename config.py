import argparse 

parser = argparse.ArgumentParser(description='argparse manual')


parser.add_argument('--N_TRAIN', required=False, default=300, help='fuck')
parser.add_argument('--N_TEST', required=False, default=100, help='fuck')
parser.add_argument('--FEATURE_DIM', required=False, default=5, help='fuck')


parser.add_argument('--ENN', required=False, default = True)
# parser.add_argument('--ENN', required=False, default = False)

parser.add_argument('--N_ITEM', required=False, default = 4)
parser.add_argument('--SEED_DATA', required=False, default = 2)


parser.add_argument('--Node_Sizes', required=False, default=[5, 10, 10, 1])
parser.add_argument('--STEPS', required=False, default=10000000)
parser.add_argument('--LOSS_PRINT', required=False, default=10000)
parser.add_argument('--TEST_LOSS_PRINT', required=False, default=True)
parser.add_argument('--TEST_BATCH', required=False, default=100)


parser.add_argument('--SCALE', required=False, default=0.1)
parser.add_argument('--DROPOUT', required=False, default=0.1)


# parser.add_argument('--target', required=False, help='fuck')