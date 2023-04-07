MODEL=$1
DATASET_DIR=$2
MODE=$3

# Saliency evaluation
for DATASET in ECSSD DUTS-TEST DUT-OMRON
do
    python main_found_evaluate.py --eval-type saliency --dataset-eval $DATASET \
            --evaluation-mode $MODE --apply-bilateral --dataset-dir $DATASET_DIR
done


