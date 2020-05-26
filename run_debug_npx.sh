#!/bin/bash

export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN=500
export OMP_NUM_THREADS=1
python -m sockeye.train \
	-d /mnt/dldata/bchu/data/prepared_data_2 \
	-vs /mnt/shared/bchu/newstest2016.tc.de.bpe \
	-vt /mnt/shared/bchu/newstest2016.tc.en.bpe \
	-o debug_npx_model \
	--num-layers 6 \
	--transformer-model-size 512 \
	--transformer-attention-heads 8 \
	--transformer-feed-forward-num-hidden 2048 \
	--weight-tying-type src_trg_softmax \
	--optimizer adam \
	--batch-size 2048 \
	--update-interval 4 \
	--round-batch-sizes-to-multiple-of 8 \
	--checkpoint-interval 25 \
	--initial-learning-rate 0.0004 \
	--learning-rate-reduce-factor 0.9 \
	--learning-rate-reduce-num-not-improved 8 \
	--max-checkpoints 2 \
	--decode-and-evaluate 500 \
	--device-ids -1 \
	--no-hybridization \
	--seed 712
