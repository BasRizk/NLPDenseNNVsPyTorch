export HIDDEN_UNITS=200
export LEARNING_RATE=0.05
export NUM_EPOCHS=40
export MAX_SEQUENCE_LENGTH=20

export MINI_BATCH_SIZE=256

export DATASET=datasets/odia.train.txt
export EMBEDDING_FILE=fasttext.wiki.300d.vec,unk-odiya.vec

if [ $TORCH == true ]; 
then 
echo "using torch"
export EXEC_FILE=train-torch.py;
export OUT_MODEL_FILE=torch.odia.model
else
echo "using basic implementation"
export EXEC_FILE=train.py; 
export OUT_MODEL_FILE=odia.model
fi

python $EXEC_FILE -u $HIDDEN_UNITS -l $LEARNING_RATE -f $MAX_SEQUENCE_LENGTH -b $MINI_BATCH_SIZE -e $NUM_EPOCHS -E $EMBEDDING_FILE -i $DATASET -o $OUT_MODEL_FILE