CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#foreach(TARGET skiprnnlm attentional  biattentional test_expressions word_context hard_attentional dam)
foreach(TARGET attentional attentional_plp)
  ADD_EXECUTABLE(${TARGET} ${TARGET}.cc)
  target_link_libraries(${TARGET} cnn ${LIBS})
  if (WITH_CUDA_BACKEND)
    add_dependencies(${TARGET} cnncuda)
    target_link_libraries(${TARGET} cnncuda)
    CUDA_ADD_CUBLAS_TO_TARGET(${TARGET})
  endif (WITH_CUDA_BACKEND)
endforeach()

#ADD_EXECUTABLE(markov markov.cc corpus.cc)
#target_link_libraries(markov cnn ${LIBS})
#
#ADD_EXECUTABLE(markov-old markov-old.cc corpus_old.cc)
#target_link_libraries(markov-old cnn ${LIBS})

#ADD_EXECUTABLE(rnnlm-tensor rnnlm-tensor.cc rtnn.cc)
#target_link_libraries(rnnlm-tensor cnn ${LIBS})

#ADD_EXECUTABLE(rnnlm-tensor2 rnnlm-tensor2.cc rtnn.cc)
#target_link_libraries(rnnlm-tensor2 cnn ${LIBS})
