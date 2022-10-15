# NLPDenseNNVsPyTorch
Comparing From Scratch Feed Forward Neural Network with Pytorch implementation on 4 different NLP datasets.

## Instructions for Training
>  There is `train_DATASETNAME.sh` per each dataset for example of training.
> You may call it directly to run from scratch model or `train_DATASETNAME.sh torch` or basically any other non-empty argument to run it on pytorch implementation instead.
> Or you could follow the mentioned values here

### Parameters:
  - `-u` network description
    - Ex1 `sigmoid15,d0.5,relu10`
    - Ex2 `relu200`
    - options are `sigmoidX`, `reluX` , `dX` for dropouts, where `X` is number of hidden units or zeroing probability for dropouts
  - `-w`* weights initialization technique
    - `zeros`, `xavier`, `scaling` by 0.01, `random`, or `normal` (refer to report)
  - `-l` learning rate
  - `-f` max sequence length
  - `-b` mini-batch size
  - `-e` number of epochs to train for
  - `-E` word embedding filepath
  - `-i` training filepath
  - `-o` model filepath to be written
  - `-d` debug filepath to be written (make sure to have folder logs)
  
\* Used with from scratch implementation only


### Example
>
> EXEC_FILE = train.py
> or
> EXEC_FILE = train-torch.py 
> 
> `python $EXEC_FILE -u $HIDDEN_UNITS -l $LEARNING_RATE -f $MAX_SEQUENCE_LENGTH -b $MINI_BATCH_SIZE -e $NUM_EPOCHS -E $EMBEDDING_FILE -i $DATASET -o $OUT_MODEL_FILE -w $WEIGHTS_INIT -d $DEBUG_FILE`



## Instructions for Classifying
### Parameters:
  - `-m` model filename (either start with `pytorch` or without)
  - `-i` test data-set relative filepath
  - `-o` output (inference) desired relative filepath
  
  
### Example
>
> EXEC_FILE = train.py
> or
> EXEC_FILE = train-torch.py 
> 
> `python $EXEC_FILE -m nb.4dim.model -i 4dim.sample.txt -o 4dim.out.txt`