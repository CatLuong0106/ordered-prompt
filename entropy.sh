MAIN_DIR=$(pwd)
DATASET=agnews
LOGDIR=experiment/$DATASET;
SAMPLE_MODE=balance
NSHOT=4
SEED=1
MODEL=llama3
python entropy.py --true $LOGDIR/true_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.pkl \
                    --fake $LOGDIR/fake_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.pkl \
                    --topk 4 --save $LOGDIR/result_${DATASET}_${NSHOT}_shot_${MODEL}_seed${SEED}.json