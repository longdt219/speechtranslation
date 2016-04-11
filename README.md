# Speech Translation/Alignment
This is the implementation of our NAACL paper titled. 
``An Attentional Model for Speech Translation Without Transcription''

#### Getting started

This implementation is based on C++ neural network library (CNN) with development version of Eigen. 

    hg clone https://bitbucket.org/eigen/eigen/ 
    git clone https://github.com/longdt219/speechtranslation.git
    cd speechtranslation	
    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=../eigen
    make -j 10

## Experiment with Phone - Word 
This is experiment directly from phone sequence to word. Similar with machine translation problem. 
The data format is 
```<s> source phone </s> ||| <s> target words </s>``` 
We attached a tiny training data for the demo purposes. 
#### Train the attentional model
Use the following script 
```
./build/attentional_model/attentional --train data/train.attentional --devel data/dev.attentional --lstm --bidirectional -a 32 --hidden 32 --parameters model.phone --epochs 50 --coverage 0.05 --trainer sgd --layers 4 --giza --smoothsm 0.1
```
Some options :
- parameters: periodically save the parameters to this file so that learning can be resume
- lstm: use LSTM for RNN (other options are: GRU and RNN) where GRU use Gated-Recurrent Unit
- coverage: use the coverage penalty described in the paper
- layers n: stack n layers of lstm on the target
- giza: use giza features described in the paper
- smoothsm: use smoothing softmax function described in the paper
- help: display the detail of other options.

#### Output the translation and retrieval task
We need to initialise with the trained model and use the test data instead of dev data. 
```
./build/attentional_model/attentional --train data/train.attentional --devel data/test.attentional --lstm --bidirectional -a 32 --hidden 32 --initialise model.phone --epochs 50 --coverage 0.05 --trainer sgd --layers 4 --giza --smoothsm 0.1 --translation
```
The output will be the translation on test, first 200 sentences of train and some output for retrieval task. 

#### Use the attentinal model for reranker 
Need to extract the first 100 hypothesis from Moses, assuming in the file `data/rescore.pairs` 
Run the following script
```
./build/attentional_model/attentional --train data/train.attentional --devel data/test.attentional --lstm --bidirectional -a 32 --hidden 32 --initialise model.phone --epochs 50 --coverage 0.05 --trainer sgd --layers 4 --giza --smoothsm 0.1 --rescore --test data/rescore.pairs
```
The model will score each pair and add the score at the end. The final translation will be the candidate having **lowest** score. 
## Experiment with Speech - Word 
#### Extract speech features 
We use [SPRACHcore] (http://www1.icsi.berkeley.edu/~dpwe/projects/sprach/sprachcore.html) to extract plp features from speech file with the following options.
    `./feacalc -hpfilter 100 -dither -domain cepstra -deltaorder 2 -plp 12 -sr 16000 -opformat ascii -o OUTPUTFILE INPUTFILE`
Obviously, the sample rate (`-sr`) will be different based on your data. 

#### Training the model directly from speech signal 
For demo, I added some tiny data extracted from TIMIT in `data` folder
Run the following 
```
./build/attentional_model/attentional_plp --ttrain data/text/ --strain data/plp/ --lstm --bidirectional --align 32 --hidden 32 --parameters model.speech.plp --epochs 50 --coverage 0.05 --trainer sgd --layers 4 --giza --pyramid --smoothsm 0.1 --split data.split
```
Some options : 
- strain: source folder storing all plp files. One plp file represent one speech sentence.
- ttrain: target folder where each file is a translation or transcription of the speech sentence. Note that files in strain and ttrain should have the same ID. 
- pyramid: use the pyramidal structure described in the paper 
- split: a file specify the data split. It will have 3 lines, each line specify list of files for Train, Dev and Test  

#### Testing 
Show the translation from the trained model for test data (and some train data).  
```
./build/attentional_model/attentional_plp --ttrain data/text/ --strain data/plp/ --lstm --bidirectional --align 32 --hidden 32 --initialise model.speech.plp --epochs 50 --coverage 0.05 --trainer sgd --layers 4 --giza --pyramid --smoothsm 0.1 --split data.split --translation
```

### Others
There are several useful debugging/outputting options for example 
-display: to shows the alignment matrix in `tikz` format which can be imported to `(la)tex` 
-verbose: output intermediate alignment/translation.  

