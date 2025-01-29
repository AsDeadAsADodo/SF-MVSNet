#!/usr/bin/env bash
TESTPATH="/root/autodl-tmp/tankandtemples/advanced" # path to dataset
TESTLIST="lists/tnt/adv.txt"
NORMAL_PATH="/root/autodl-tmp/highresTNT/" 												
CKPT_FILE="./xoutputs/dtu_training/model_000011.ckpt" 		    # path to checkpoint
OUTDIR="/root/autodl-tmp/outputs/tnt_testing_Diff/" 									# path to save the results
if [ ! -d $OUTDIR ]; then
	mkdir -p $OUTDIR
fi

CUDA_VISIBLE_DEVICES=0 nohup python test.py \
--dataset=tnt_eval \
--num_view=11 \
--batch_size=1 \
--normalpath=$NORMAL_PATH \
--interval_scale=1.0 \
--numdepth=192 \
--ndepths="48,32,8"  \
--depth_inter_r="4,1,0.25" \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--outdir=$OUTDIR  \
--filter_method="dynamic" \
--loadckpt $CKPT_FILE ${@:2} | tee -a $LOG_DIR/log.txt > adv.log 2>&1 &

#Using this script to generate depth maps and then run the dynamic_fusion.sh to generate the final point cloud.

