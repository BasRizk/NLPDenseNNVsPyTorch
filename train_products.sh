export HIDDEN_UNITS=300
export LEARNING_RATE=0.05
export NUM_EPOCHS=1000
export MAX_SEQUENCE_LENGTH=10

export MINI_BATCH_SIZE=256

export DATASET=datasets/products.train.txt
export EMBEDDING_FILE=glove.6B.50d.txt,unk-eng.vec

if [ $TORCH == true ]; 
then 
echo "using torch"
export EXEC_FILE=train-torch.py;
export OUT_MODEL_FILE=torch.products.model
else
echo "using basic implementation"
export EXEC_FILE=train.py; 
export OUT_MODEL_FILE=products.model
fi

python $EXEC_FILE -u $HIDDEN_UNITS -l $LEARNING_RATE -f $MAX_SEQUENCE_LENGTH -b $MINI_BATCH_SIZE -e $NUM_EPOCHS -E $EMBEDDING_FILE -i $DATASET -o $OUT_MODEL_FILE