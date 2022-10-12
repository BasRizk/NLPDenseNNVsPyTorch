export HIDDEN_UNITS=100
export LEARNING_RATE=0.05
export NUM_EPOCHS=50
export MAX_SEQUENCE_LENGTH=10

export MINI_BATCH_SIZE=256

export DATASET=datasets/questions.train.txt
export EMBEDDING_FILE=ufvytar.100d.txt,unk-ufvytar.vec

if [ $TORCH == true ]; 
then 
echo "using torch"
export EXEC_FILE=train-torch.py;
export OUT_MODEL_FILE=torch.questions.model
else
echo "using basic implementation"
export EXEC_FILE=train.py; 
export OUT_MODEL_FILE=questions.model
fi

python $EXEC_FILE -u $HIDDEN_UNITS -l $LEARNING_RATE -f $MAX_SEQUENCE_LENGTH -b $MINI_BATCH_SIZE -e $NUM_EPOCHS -E $EMBEDDING_FILE -i $DATASET -o $OUT_MODEL_FILE