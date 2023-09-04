#!/bin/bash

# SIMPLY EXECUTE bash run.sh

cat $0

EXTERNAL_MAX_TRAIN_SIZE=$1
DATASET=$2
EXISTING_MODEL_PATH=$3
#EXTERNAL_NUM_RUN=$4 # <-- E' utile quando si riesegue solo il test su un modello gia' addestrato


BERT_MODELS_DIR=pre-trained_models
OUTPUT_DIR=./output
EPOCHS=1 # to do: CHANGE TO 4!
BATCH_SIZE=6 
SEQ_LENGTH=300 #250 ## mak num tokens in sentence (max 512)
TIMESTAMP=`date +"%Y%m%d_%H%M%S"`
DATETIME=`date +"%Y-%m-%dT%H:%M:%S"`
export BERT_MODELS_DIR
TASKS=("scholarly") # BERT-MLP
#TASKS=("scholarly_VANILLA") # BERT-VANILLA
#TASKS=("scholarly_NO_TEXT") # USE ONLY EMBEDDINGS VECTORS WITHOUT TEXTS
DATA_DIR="./scholarly_bert_experiments/bert_input_data"
TARGETS=("scholarly")  # 1_topic_MIN_0.7_MAX_1.0

NUM_RUN=4 #$EXTERNAL_NUM_RUN
FIRST_RUN=1 #$EXTERNAL_NUM_RUN

#MAX_TRAIN_SIZE=3000
MAX_TRAIN_SIZE=$EXTERNAL_MAX_TRAIN_SIZE
MAX_DEV_SIZE=1500
MAX_TEST_SIZE=1500
LOOKUP_SUFFIX=_entities.pickle
VECTORTYPE="sentence_embedding"

RANDOM_SEED=42
if [ -z ${SEQ_LENGTH+x} ]; then SEQ_LENGTH=512; fi

for MAX_TRAIN_SIZE in `seq 3000 3000 21000`;do
    MAX_DEV_SIZE=1500 #$(($MAX_TRAIN_SIZE/5))
    MAX_TEST_SIZE=1500 #$(($MAX_TRAIN_SIZE/5))
    for TASK_NAME in ${TASKS[@]};do
      rm -rf $OUTPUT_DIR/${TASK_NAME}/
      TASK_EXECUTION_DIR=${TASK_NAME}__${DATASET}__train-${MAX_TRAIN_SIZE} #__execution-${TIMESTAMP}
      mkdir $OUTPUT_DIR/$TASK_EXECUTION_DIR

      for t in ${TARGETS[@]};do 
        LOOKUP_NAME=${t}$LOOKUP_SUFFIX
        for i in `seq ${FIRST_RUN} ${NUM_RUN}`;do

          EXPERIMENT_DIR=${TASK_NAME}_${t}_${DATASET}_${TIMESTAMP}__trainsize-${MAX_TRAIN_SIZE}__epocs-${EPOCHS}__run-${i}
          RUN_METADATA="\"time\":\"${DATETIME}\", \"target\": \"${t}\", \"task\":\"${TASK_NAME}\",\"epochs\":${EPOCHS},\"max_train_size\":\"${MAX_TRAIN_SIZE}\",\"vector_type\":\"${VECTORTYPE}\",\"tot_runs\":${NUM_RUN},\"run\":${i},\"random_seed\": \"${RANDOM_SEED}\", \"kg\": \"n/a\",\"path\":\"$OUTPUT_DIR/$TASK_EXECUTION_DIR/$EXPERIMENT_DIR\""
          #RUN_METADATA="\"time\":\"${DATETIME}\", \"target\": \"${t}\", \"task\":\"${TASK_NAME}\",\"epochs\":${EPOCHS},\"max_train_size\":\"${MAX_TRAIN_SIZE}\",\"vector_type\":\"n/a\",          \"tot_runs\":${NUM_RUN},\"run\":${i}, \"random_seed\": \"${RANDOM_SEED}\", \"kg\": \"${KG_NAME}\",\"path\":\"$OUTPUT_DIR/$TASK_EXECUTION_DIR/$EXPERIMENT_DIR\""
          echo $RUN_METADATA 
          echo "Lookup file: $LOOKUP_NAME"
          echo "Vectory type: $VECTORTYPE"

          #debugpy-run cli.py \
          python -u cli.py \
                run_on_val_and_test \
                ${TASK_NAME} \
                0 \
                ./extras/ \
                ${DATA_DIR}/train_${MAX_TRAIN_SIZE}.pickle \
                ${DATA_DIR}/dev.pickle \
                ${DATA_DIR}/test.pickle \
                $OUTPUT_DIR \
                --epochs $EPOCHS --random_state ${RANDOM_SEED} \
                --batch_size $BATCH_SIZE --seq_length ${SEQ_LENGTH} \
                --max_training_size $MAX_TRAIN_SIZE \
                --max_dev_size $MAX_DEV_SIZE \
                --max_test_size $MAX_TEST_SIZE \
                --lookup_name $LOOKUP_NAME \
		--mkdir True \
		--just_validate False #\
		#--existing_model_path $EXISTING_MODEL_PATH

                #${DATA_DIR}/train_${MAX_TRAIN_SIZE}_${t}_${DATASET}.pickle \
                #${DATA_DIR}/dev_${t}_${DATASET}.pickle \
                #${DATA_DIR}/test_${t}_${DATASET}.pickle \
	  
          echo $RUN_METADATA 
          cat $OUTPUT_DIR/${TASK_NAME}/test_report.txt
          
          mv $OUTPUT_DIR/${TASK_NAME} $OUTPUT_DIR/$TASK_EXECUTION_DIR/$EXPERIMENT_DIR_${MAX_TRAIN_SIZE}_${i}
          echo "{ ${RUN_METADATA} }" > $OUTPUT_DIR/$TASK_EXECUTION_DIR/$EXPERIMENT_DIR_${MAX_TRAIN_SIZE}_${i}/run_metadata.json

          #mv $OUTPUT_DIR/${TASK_NAME} $OUTPUT_DIR/$TASK_EXECUTION_DIR/$EXPERIMENT_DIR_${EXTERNAL_MAX_TRAIN_SIZE}_${i}
          #echo "{ ${RUN_METADATA} }" > $OUTPUT_DIR/$TASK_EXECUTION_DIR/$EXPERIMENT_DIR_${EXTERNAL_MAX_TRAIN_SIZE}_${i}/run_metadata.json	  
      
        done
      done
    done
done
