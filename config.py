import argparse 

parser = argparse.ArgumentParser(description='argparse manual')


parser.add_argument('--N_TRAIN', required=False, type=int, default=300, help='fuck')
parser.add_argument('--N_TEST', required=False, type=int, default=300, help='fuck')
parser.add_argument('--FEATURE_DIM', required=False, type=int, default=5, help='fuck')


parser.add_argument('--ENN', required=False, type=int, default = 1) # 0: no enn, 1: enn 
# parser.add_argument('--ENN', required=False, default = False)

parser.add_argument('--N_ITEM', required=False, type=int, default = 4)
parser.add_argument('--SEED_DATA', required=False, type=int, default = 0)
parser.add_argument('--SEED_TRAIN', required=False, type=int, default = 0)



parser.add_argument('--Node_Sizes', required=False, default=[5, 10, 10, 1])
# parser.add_argument('--Diag_Sizes', required=False, default=[5, 2, 2, 1])


parser.add_argument('--SUBNODE', required=False, type=int, default=2)

parser.add_argument('--ITERATIONS', required=False, type=int, default=1000)
parser.add_argument('--LOSS_PRINT', required=False, type=int, default=100)
parser.add_argument('--TEST_LOSS_PRINT', required=False, type=int, default=True)
parser.add_argument('--TEST_BATCH', required=False, type=int, default=100)

parser.add_argument('--LR', required=False, type=float, default=0.0001)
parser.add_argument('--SCALE', required=False, type=float, default=0.1)
parser.add_argument('--DROPOUT', required=False, type=float, default=0.1)
# parser.add_argument('--DROPOUT', required=False, default=0.1)



# parser.add_argument('--target', required=False, help='fuck')