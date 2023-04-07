MODEL=$1
DATASET_DIR=$2

# Unsupervised object discovery evaluation
for DATASET in VOC07 VOC12 COCO20k
do
    python main_found_evaluate.py --eval-type uod --dataset-eval VOC07 \
            --evaluation-mode single --dataset-dir $DATASET_DIR
done


