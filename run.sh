#!/bin/bash
ROOT=/media/cxh/lycan
#SCHEME=mkl_csrmm
#SCHEME=mkl_gemm
#SCHEME=gpu_gemm
SCHEME=gpu_csrmm
PROTO=/$ROOT/caffe/models/bvlc_reference_caffenet/train_val.prototxt
WEIGHTS=/$ROOT/SkimCaffe/models/bvlc_reference_caffenet/logs/acc_57.5_0.001_5e-5_ft_0.001_5e-5/0.001_5e-05_0_1_0_0_0_0_Sun_Jan__8_07-35-54_PST_2017/caffenet_train_iter_640000.caffemodel
OUT=/$ROOT/logs/$SCHEME-caffenet.log
MODE="-gpu 0"

echo ">>>Start run $SCHEME at `date`" > $OUT 
./build/tools/caffe.bin test -model $PROTO -weights $WEIGHTS -gpu 0 2>> $OUT
echo ">>> End run $SCHEME at `date`" >> $OUT 
