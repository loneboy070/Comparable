#!/bin/bash
STEPS=10000000
LOSS_PRINT=20000
N_TRAIN=300
N_TEST=100
ENN=false
TEST_BATCH=100
N_ITEM=10
SEED_DATA=0
LR=0.0003

for STEPS in 10000000
do
    for N_TRAIN in 111
    do
        for N_ITEM in 2
        do
            for SEED_DATA in 0
            do
                for N_TEST in 100
                do
                    for ENN in false
                    do
                        sleep 3
                        python train.py --STEPS $STEPS --LOSS_PRINT $LOSS_PRINT --N_TRAIN $N_TRAIN --N_TEST $N_TEST --ENN $ENN --TEST_BATCH $TEST_BATCH --N_ITEM $N_ITEM --SEED_DATA $SEED_DATA --LR $LR &
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
