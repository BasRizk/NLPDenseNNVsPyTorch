export HIDDEN_UNITS=100
export LEARNING_RATE=0.05
export NUM_EPOCHS=20
export MAX_SEQUENCE_LENGTH=20

export MINI_BATCH_SIZE=8
export EMBEDDING_FILE=ufvytar.100d.txt,unk-ufvytar.vec

export DT_NAME=questions
export DATASET=datasets/$DT_NAME.train.txt
echo "Training on" $DATASET

if [[ ! -z ${TORCH+x} ]] && [[ $TORCH == true ]]; 
then 
echo "using torch"
export EXEC_FILE=train-torch.py
export OUT_MODEL_FILE=torch.$DT_NAME.model
else
echo "using basic implementation"
export EXEC_FILE=train.py
export OUT_MODEL_FILE=$DT_NAME.model
fi

export DEBUG_FILE=logs/${OUT_MODEL_FILE}_${HIDDEN_UNITS}_${LEARNING_RATE}_${MAX_SEQUENCE_LENGTH}_${MINI_BATCH_SIZE}_${NUM_EPOCHS}.debug

python $EXEC_FILE -u $HIDDEN_UNITS -l $LEARNING_RATE -f $MAX_SEQUENCE_LENGTH -b $MINI_BATCH_SIZE -e $NUM_EPOCHS -E $EMBEDDING_FILE -i $DATASET -o $OUT_MODEL_FILE -d $DEBUG_FILE