#!/usr/bin/env bash
TESTPATH="data/DTU/mvs_testing/dtu" 						# path to dataset dtu_test
#TESTLIST="lists/dtu/test.txt"
TESTLIST="lists/dtu/test1.txt"
CKPT_FILE="checkpoints/model_dtu.ckpt"			   # path to checkpoint file, you need to use the model_dtu.ckpt for testing
FUSIBLE_PATH="" 								 	# path to fusible of gipuma
OUTDIR="outputs/dtu_testing" 						  # path to output
if [ ! -d $OUTDIR ]; then
	mkdir -p $OUTDIR
fi


python test.py \
--dataset=general_eval \
--batch_size=1 \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--loadckpt=$CKPT_FILE \
--outdir=$OUTDIR \
--numdepth=512 \
--ndepths="48,32,8" \
--depth_inter_r="4.0,1.0,0.5" \
--interval_scale=1.06 \
--filter_method="normal" \
--fusibile_exe_path=$FUSIBLE_PATH
#--filter_method="normal"

#--dataset=general_eval
#--batch_size=1
#--testpath="data/DTU/mvs_testing/dtu"
#--testlist="lists/dtu/test1.txt"
#--loadckpt="checkpoints/model_dtu.ckpt"
#--outdir="outputs/dtu_testing"
#--numdepth=192
#--ndepths="48,32,8"
#--depth_inter_r="4.0,1.0,0.5"
#--interval_scale=1.06
#--filter_method="normal"