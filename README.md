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


#### Extract speech features 
We use [SPRACHcore] (http://www1.icsi.berkeley.edu/~dpwe/projects/sprach/sprachcore.html) to extract plp features from speech file with the following options.
    ./feacalc -hpfilter 100 -dither -domain cepstra -deltaorder 2 -plp 12 -sr 16000 -opformat ascii -o OUTPUTFILE INPUTFILE
Obviously, the sample rate (-sr) will be different based on your data. 

#### Training the model directly from speech signal 
Run the following 
```shell
./build/attentional_plp --ttrain ../../data/TEDTALKS/fr_translation_unk/ --strain ../../data/TEDTALKS/fr_plp39/ --lstm --bidirectional -a 256 --hidden 256 --parameters model.speech.plp --epochs 50 --coverage 0.05 --trainer sgd --layers 4 --giza --pyramid --smoothsm 0.1 --split data.split 2> log.plp39.word.unk.pyramid.256.translation
```
Where : 
    strain: source folder storing all plp files. One plp file represent one speech sentence. 
    

