#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV3 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./scripts/finetune_mobilenet_v1_on_brilliance.sh
set -e

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/tmp/brilliance/pretrain

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/brilliance/train

# Where the dataset is saved to.
DATASET_DIR=/tmp/brilliance/dataset

# Output graph 
OUTPUT_GRAPH=/tmp/brilliance/output/mobilenet_v1_224.pb

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/mobilenet_v1_1.0_224_frozen.pb ]; then
  wget http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz
  tar -xvf mobilenet_v1_1.0_224.tgz -C ${PRETRAINED_CHECKPOINT_DIR}
  rm mobilenet_v1_1.0_224.tgz
  echo 'model_checkpoint_path: "'${PRETRAINED_CHECKPOINT_DIR}'/mobilenet_v1_1.0_224.ckpt"' > checkpoint
  mv checkpoint ${PRETRAINED_CHECKPOINT_DIR}
fi

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=brilliance \
  --dataset_dir=${DATASET_DIR}

# Fine-tune only the new layers for 1000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=brilliance \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v1 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR} \
  --checkpoint_exclude_scopes=MobilenetV1/Logits \
  --trainable_scopes=MobilenetV1/Logits \
  --max_number_of_steps=1000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 \
  --clone_on_cpu
#  --save_interval_secs=60 \
#  --save_summaries_secs=60 \

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=brilliance \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v1 \
  --clone_on_cpu

# Fine-tune all the new layers for 500 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=brilliance \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v1 \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=500 \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 \
  --clone_on_cpu
#  --save_interval_secs=60 \
#  --save_summaries_secs=60 \

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=brilliance \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v1 \
  --clone_on_cpu

# Export inference graph
python export_inference_graph.py \
  --alsologtostderr \
  --model_name=mobilenet_v1 \
  --output_file=${OUTPUT_GRAPH} \
  --dataset_name=brilliance \
  --dataset_dir=${DATASET_DIR} \
  --image_size=224

