#!/bin/bash
#examples
#build/tools/caffe.bin test -model models/bvlc_reference_caffenet/train_val.prototxt -weights models/bvlc_reference_caffenet/caffenet_train_iter_640000.caffemodel -gpu 0 -conv_mode 1 -iterations 3
#build/tools/caffe.bin test -model models/bvlc_googlenet/train_val.prototxt -weights models/bvlc_googlenet/gesl_0.686639_0.001_0.00005_ft_0.001_0.0001.caffemodel -gpu 0 -conv_mode 2 -iterations 9
#build/tools/caffe.bin test -model models/resnet/ResNet-50-train-val-gesl.prototxt -weights models/resnet/caffenet_train_iter_2000000.caffemodel -gpu 0 -conv_mode 0 -iterations 6

#ROOT=/media/cxh/lycan
#SCHEME=mkl_csrmm
#SCHEME=mkl_gemm
#SCHEME=gpu_gemm
#SCHEME=gpu_csrmm
SCHEME=cpu_sconv
PROTO=./models/bvlc_reference_caffenet/train_val.prototxt
WEIGHTS=../SkimCaffe/models/bvlc_reference_caffenet/logs/acc_57.5_0.001_5e-5_ft_0.001_5e-5/0.001_5e-05_0_1_0_0_0_0_Sun_Jan__8_07-35-54_PST_2017/caffenet_train_iter_640000.caffemodel
OUT=../logs/$SCHEME-caffenet.log
ERR=../logs/$SCHEME-more.log
#MODE="-gpu 0"
export GLOG_minloglevel=2
echo ">>>Start run $SCHEME at `date`" > $OUT 
./build/tools/caffe.bin test -model $PROTO -weights $WEIGHTS $MODE 1>> $OUT 2> $ERR
echo ">>> End run $SCHEME at `date`" >> $OUT 
