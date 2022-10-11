export HIDDEN_UNITS=100
export LEARNING_RATE=0.05
export NUM_EPOCHS=50
export MAX_SEQUENCE_LENGTH=10

export MINI_BATCH_SIZE=256

export DATASET=datasets/questions.train.txt
export OUT_MODEL_FILE=questions.model
export EMBEDDING_FILE=ufvytar.100d.txt,unk-ufvytar.vec

python train.py -u $HIDDEN_UNITS -l $LEARNING_RATE -f $MAX_SEQUENCE_LENGTH -b $MINI_BATCH_SIZE -e $NUM_EPOCHS -E $EMBEDDING_FILE -i $DATASET -o $OUT_MODEL_FILE