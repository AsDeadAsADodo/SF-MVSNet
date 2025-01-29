#!/usr/bin/env bash
TESTPATH="/root/autodl-tmp/dtu_test" 						# path to dataset dtu_test
TESTLIST="lists/dtu/test.txt"							# path to data_list
NORMALPATH="/root/autodl-tmp/highresdtutest/"							# path to normal map
CKPT_FILE="xoutputs/dtu_training/model_000011.ckpt"	    # path to checkpoint file
FUSIBLE_PATH="/root/fusibile/fusibile"	  # path to fusible of gipuma
OUTDIR="/root/autodl-tmp/outputs/dtu_test" 						  # path to output
if [ ! -d $OUTDIR ]; then
	mkdir -p $OUTDIR
fi


CUDA_VISIBLE_DEVICES=0 python test.py \
--dataset=general_eval \
--batch_size=1 \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--normalpath=$NORMALPATH \
--loadckpt=$CKPT_FILE \
--outdir=$OUTDIR \
--numdepth=192 \
--ndepths="48,32,8" \
--depth_inter_r="4.0,1.0,0.5" \
--interval_scale=1.06 \
--filter_method="o3d" \
--fusibile_exe_path=$FUSIBLE_PATH

