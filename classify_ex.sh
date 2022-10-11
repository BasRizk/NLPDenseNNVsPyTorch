export TEST_FILE=4dim.sample.txt
export MODEL_FILE=nn.4dim.model
export OUTPUT_FILE=sample.4dim.out

python classify.py -m $MODEL_FILE -i $TEST_FILE -o $OUTPUT_FILE