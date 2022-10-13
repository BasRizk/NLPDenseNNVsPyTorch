export HIDDEN_UNITS=d\(0.4\),500
export LEARNING_RATE=0.05
export NUM_EPOCHS=150
export MAX_SEQUENCE_LENGTH=500

export MINI_BATCH_SIZE=128
export EMBEDDING_FILE=glove.6B.50d.txt,unk-eng.vec

export DT_NAME=4dim
export DATASET=datasets/$DT_NAME.train.txt
echo "Training on" $DATASET

echo $TORCH Okay
if [[ ! -z ${TORCH+x} ]] && [[ $TORCH == true ]]; 
then 
echo "using torch"
export EXEC_FILE=train-torch.py
export OUT_MODEL_FILE=torch.$DT_NAME.model
else
echo "using basic implementation"
export EXEC_FILE=train.py; 
export OUT_MODEL_FILE=$DT_NAME.model
fi

export DEBUG_FILE=logs/${OUT_MODEL_FILE}_${HIDDEN_UNITS}_${LEARNING_RATE}_${MAX_SEQUENCE_LENGTH}_${MINI_BATCH_SIZE}_${NUM_EPOCHS}.debug

python $EXEC_FILE -u $HIDDEN_UNITS -l $LEARNING_RATE -f $MAX_SEQUENCE_LENGTH -b $MINI_BATCH_SIZE -e $NUM_EPOCHS -E $EMBEDDING_FILE -i $DATASET -o $OUT_MODEL_FILE -d $DEBUG_FILE