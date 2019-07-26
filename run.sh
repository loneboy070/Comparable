#!/bin/bash
ITERATIONS=120000
LOSS_PRINT=400
N_TRAIN=300
N_TEST=100
ENN=0
TEST_BATCH=300
N_ITEM=10
SEED_DATA=0
LR=0.0001
START_ITER=120000
for END_ITER in 300000
do
    for SUBNODE in 2
    do
        for N_ITEM in 8
        do
            for SEED_DATA in 0
            do
                for SEED_TRAIN in  0 1 2 3 4
                do
                    for N_TEST in 100
                    do
                        for ENN in 0
                        do
                            sleep 3
                            python train.py --ITERATIONS $ITERATIONS --LOSS_PRINT $LOSS_PRINT --N_TRAIN $N_TRAIN --N_TEST $N_TEST --ENN $ENN --TEST_BATCH $TEST_BATCH --N_ITEM $N_ITEM --SEED_DATA $SEED_DATA --LR $LR --SUBNODE $SUBNODE --SEED_TRAIN $SEED_TRAIN --START_ITER $START_ITER --END_ITER $END_ITER &


                            ret=$?
                            if [ $ret -ne 0 ]; 
                            then
                            #Handle failure
                            #exit if required
                                exit 1
                            fi

                        done
                    done
                done
            done
        done
    done
done