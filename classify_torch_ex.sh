export TEST_FILE=4dim.sample.txt
export MODEL_FILE=torch.4dim.model
export OUTPUT_FILE=4dim.sample.torch.out

python classify-torch.py -m $MODEL_FILE -i $TEST_FILE -o $OUTPUT_FILE