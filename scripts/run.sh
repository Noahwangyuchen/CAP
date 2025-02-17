for dataset_dir in 'data_' ; do # add here the path to the folder containing dataset folders
for vis_encoder in 'ViT-B/32'; do
for split_seed in 500; do
for dataset_name in EuroSAT; do # Choose among: RESICS45 DTD Flowers102 EuroSAT FGVCAircraft CUB
for optim_seed in 1; do #
for lr in 0.01; do
for margin_scale in 12.0; do
    
    export OPTIM_SEED="$optim_seed"
    export VIS_ENCODER="$vis_encoder"
    export DATASET_NAME="$dataset_name"
    export SPLIT_SEED="$split_seed"
    export DATASET_DIR="$dataset_dir"
    export MARGIN_SCALE="$margin_scale"
    export LR="$lr"
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file methods_config/accelerate_config.yml run_main.py --model_config ssl_config.yml # Choose among ul, ssl, and trzsl

done
done
done
done
done
done
done
