# Speech Translation/Alignment
This is the implementation of our NAACL paper titled. 
An Attentional Model for Speech Translation Without Transcription

#### Getting started

This implementation is based on C++ neural network library (CNN) with development version of Eigen. 

    hg clone https://bitbucket.org/eigen/eigen/ 
    git clone https://github.com/longdt219/speechtranslation.git

And build the code 
 
    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=../eigen
    make -j 2


#### Training Models

