export HIDDEN_UNITS=200
export LEARNING_RATE=0.05
export NUM_EPOCHS=40
export MAX_SEQUENCE_LENGTH=20

export MINI_BATCH_SIZE=256

export DATASET=datasets/odia.train.txt
export OUT_MODEL_FILE=odia.model
export EMBEDDING_FILE=fasttext.wiki.300d.vec,unk-odiya.vec

python train.py -u $HIDDEN_UNITS -l $LEARNING_RATE -f $MAX_SEQUENCE_LENGTH -b $MINI_BATCH_SIZE -e $NUM_EPOCHS -E $EMBEDDING_FILE -i $DATASET -o $OUT_MODEL_FILE