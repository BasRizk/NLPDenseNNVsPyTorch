export TEST_FILE=4dim.sample.txt
export MODEL_FILE=4dim.model
export OUTPUT_FILE=4dim.sample.out

python classify.py -m $MODEL_FILE -i $TEST_FILE -o $OUTPUT_FILE