#!/bin/bash
CUDA_DEVICE_NUM=0
BATCH_SIZE=6
OUTPUT_DIR=./output_k-bert
EPOCHS=1
TIMESTAMP=`date +"%Y%m%d_%H%M"`
DATETIME=`date +"%Y-%m-%dT%H:%M"`
#TASKS=( "airbnb__k-bert__with_vm" )
#TASKS=( "airbnb__k-bert__no_vm" )
#TASKS=( "scholarly__k-bert__no_vm" )
TASKS=( "scholarly__k-bert__with_vm" )
#DATA_DIR="./datasets/english_datasets/airbnb/"
DATA_DIR="./datasets/scholarly_dataset/"
TARGETS=("label")
#TARGETS=("avail365" "num_reviews" "rev_score" "price") 
#TARGETS=("avail365" "num_reviews" "rev_score") 
#TARGETS=( "rev_score" "price") 
#TARGETS=( "price" ) 
#DATASET="airbnb_london_20220910"
DATASET="scholarly"
NUM_RUN=1
FIRST_RUN=1
# MAX_TRAIN_SIZE=4000
# MAX_DEV_SIZE=1000
# MAX_TEST_SIZE=1000
MAX_TRIPLES_FOR_ENTITY=2
LEARNING_RATE="5e-06"  ## K-bert needs a lower learning rate than usual

SEQ_LENGTH=250 #250 ## number of tokens for each sentence, use 200 with NoKG
#KG_NAME=NoKG.spo ## Knowlege graph with no triples. used to test plain BERT in conjunction with --no_vm
#KG_NAME=KG_from_ScholarlyDataset_and_CSOclassifier_for_KBert.spo
KG_NAME=KG_3cols.spo
#VM_OPTION="--no_vm" ## don't use visible matrix

#SEQ_LENGTH=512 #number of tokens for each sentence use 512 when applying triple ingestion because the texts length will increase (max = 512 for Bert limitations)
#KG_NAME=airbnb_london_20220910.spo
#VM_OPTION=""  ## use visible matrix if you inject triples in texts

KG_PATH="brain/kgs/${KG_NAME}"

RANDOM_SEED=200  ## set random seed or comment this line if you want a different random seed for each experiment execution below

echo "Starting..."
#for MAX_TRAIN_SIZE in 9000;do
#for MAX_TRAIN_SIZE in `seq 9000 3000 12000`;do
#for MAX_TRAIN_SIZE in `seq 3000 3000 12000`;do
for MAX_TRAIN_SIZE in 3000;do
    MAX_DEV_SIZE=1500 #$(($MAX_TRAIN_SIZE/5))
    MAX_TEST_SIZE=1500 #$(($MAX_TRAIN_SIZE/5))

    for TASK_NAME in ${TASKS[@]};do
        if [ -d "$OUTPUT_DIR/${TASK_NAME}/" ]; then
            rm -rf $OUTPUT_DIR/${TASK_NAME}/ ## remove previous task execution temporary dir
        fi
        TASK_EXECUTION_DIR=${TASK_NAME}__train-${MAX_TRAIN_SIZE}__execution-${TIMESTAMP}__seq_length-${SEQ_LENGTH}__${KG_NAME}
        #mkdir $OUTPUT_DIR/$TASK_EXECUTION_DIR

        for t in ${TARGETS[@]};do 
            TRAIN_PATH="${DATA_DIR}${DATASET}__${MAX_TRAIN_SIZE}__train.tsv"
            DEV_PATH="${DATA_DIR}${DATASET}__${MAX_TRAIN_SIZE}__dev.tsv"
            TEST_PATH="${DATA_DIR}${DATASET}__${MAX_TRAIN_SIZE}__test.tsv"
            TRAIN_ENTITY_MAPPING="${DATA_DIR}${DATASET}__${MAX_TRAIN_SIZE}__train__entity_mapping_by_sentence.parquet"
            TEST_ENTITY_MAPPING="${DATA_DIR}${DATASET}__${MAX_TRAIN_SIZE}__test__entity_mapping_by_sentence.parquet"
            DEV_ENTITY_MAPPING="${DATA_DIR}${DATASET}__${MAX_TRAIN_SIZE}__dev__entity_mapping_by_sentence.parquet"
            EXPERIMENT_NAME="${TASK_NAME}_${t}"

            for i in `seq ${FIRST_RUN} ${NUM_RUN}`;do
                if [ -z ${RANDOM_SEED+x} ]; then RANDOM_SEED=$(( $RANDOM % 50 + 43 )); fi
                
                EXPERIMENT_DIR=${TASK_NAME}_${t}_${DATASET}_${TIMESTAMP}__trainsize-${MAX_TRAIN_SIZE}__epocs-${EPOCHS}__run-${i}
                RUN_METADATA="\"time\":\"${DATETIME}\", \"target\": \"${t}\", \"task\":\"${TASK_NAME}\",\"epochs\":${EPOCHS},\"max_train_size\":\"${MAX_TRAIN_SIZE}\",\"vector_type\":\"n/a\",\"tot_runs\":${NUM_RUN},\"run\":${i}, \"random_seed\": \"${RANDOM_SEED}\", \"kg\": \"${KG_NAME}\",\"path\":\"$OUTPUT_DIR/$TASK_EXECUTION_DIR/$EXPERIMENT_DIR\""

                echo $RUN_METADATA 
                echo "Random: $RANDOM_SEED"
                mkdir $OUTPUT_DIR/${TASK_NAME}/ ##temporary dir for task execution

                CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_NUM}" nohup python -u run_kbert_cls.py \
                    --experiment_name $OUTPUT_DIR/${TASK_NAME}/${EXPERIMENT_NAME} \
                    --pretrained_model_path ./models/UER_bert_base_en_uncased_model.bin \
                    --config_path ./models/google_config.json \
                    --vocab_path ./models/bert-base-uncased_google_vocab.txt \
                    --train_path ${TRAIN_PATH} \
                    --dev_path ${DEV_PATH} \
                    --test_path ${TEST_PATH} \
                    --train_entity_mapping ${TRAIN_ENTITY_MAPPING} \
                    --test_entity_mapping ${TEST_ENTITY_MAPPING} \
                    --dev_entity_mapping ${DEV_ENTITY_MAPPING} \
                    --epochs_num ${EPOCHS} --batch_size ${BATCH_SIZE} --kg_name ${KG_PATH} --workers_num 1 \
                    --train_max_size $MAX_TRAIN_SIZE --seq_length ${SEQ_LENGTH} \
                    --dev_max_size $MAX_DEV_SIZE --test_max_size $MAX_TEST_SIZE \
                    --output_model_path $OUTPUT_DIR/${TASK_NAME}/trained_model.bin \
                    --learning_rate ${LEARNING_RATE} --seed ${RANDOM_SEED} $VM_OPTION \
                    --working_dir $OUTPUT_DIR/${TASK_NAME}/ \
                    | tee $OUTPUT_DIR/${TASK_NAME}/training_output.log

                
                echo $RUN_METADATA             
                mv $OUTPUT_DIR/${TASK_NAME} $OUTPUT_DIR/$TASK_EXECUTION_DIR/$EXPERIMENT_DIR
                echo "{ ${RUN_METADATA} }" > $OUTPUT_DIR/$TASK_EXECUTION_DIR/$EXPERIMENT_DIR/run_metadata.json
            
            done
        done
    done
done
