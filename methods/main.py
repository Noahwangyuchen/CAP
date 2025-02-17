import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from logging import StreamHandler
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.stats as st
import torch
import yaml
from utils import augmentations
from accelerate import Accelerator
from requests.adapters import HTTPAdapter
from torch import nn
from urllib3.util import Retry

from data import CustomDataset, dataset_custom_prompts
from methods import (
    TextualFPL,
)
from utils import (
    Config,
    dataset_object,
    evaluate_predictions,
    get_class_names,
    get_labeled_and_unlabeled_data,
    save_parameters,
    save_predictions,
    store_results,
)

accelerator = Accelerator()
torch.set_printoptions(threshold=np.inf)

logger_ = logging.getLogger()
logger_.level = logging.INFO
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")


class AccelerateHandler(StreamHandler):
    def __init__(self, stream):
        super().__init__(stream)

    def emit(self, record):
        if accelerator.is_local_main_process:
            super().emit(record)


stream_handler = AccelerateHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger_.addHandler(stream_handler)

log = logging.getLogger(__name__)


def workflow(dataset_dir, obj_conf):
    # Get dataset name
    # We get the dataset name from the dev_config.py
    dataset = obj_conf.DATASET_NAME
    # Get class names of target task
    # define function for each dataset
    classes, seen_classes, unseen_classes = get_class_names(dataset, dataset_dir, obj_conf.SPLIT_SEED)
    # We set seen and unseen to classes, when we are not in TRZSL
    if obj_conf.PARADIGM != 'TRZSL':
        seen_classes = classes
        unseen_classes = classes
    # Create dict classes
    dict_classes = {
        "classes": classes,
        "seen_classes": seen_classes,
        "unseen_classes": unseen_classes,
    }
    label_to_idx = {c: idx for idx, c in enumerate(classes)}
    # Log number of classes
    log.info(f"\n----------------------DATA INFO-----------------------\n")
    log.info(f"Number of classes {obj_conf.SPLIT_SEED}: {len(classes)}")
    # Path for images
    data_folder = f"{dataset_dir}/{dataset}"
    log.info(f"Data folder: {data_folder}")
    log.info(f"\n-------------------------------------------------------------\n")
    
    # Get data 
    labeled_data, unlabeled_data, test_data = get_labeled_and_unlabeled_data(
        dataset, data_folder, seen_classes, unseen_classes, classes
    )

    if obj_conf.PARADIGM != 'TRZSL':
        # From labeled data of all the target classes we sample few-examples
        labeled_files, labeles = zip(*labeled_data)
        test_labeled_files, test_labeles = zip(*test_data)
        # Select few-samples
        few_shots_files = []
        few_shots_labs = []

        labeled_files = np.array(labeled_files)
        labeles = np.array(labeles)
        for c in classes:
            np.random.seed(obj_conf.validation_seed)
            indices = np.random.choice(
                np.where(labeles == c)[0], 
                size=obj_conf.N_LABEL, 
                replace=False, 
            )
            few_shots_files += list(labeled_files[indices])
            few_shots_labs += list(labeles[indices])

        log.info(f"NUMBER OF SHOTS =  {len(classes)} (NUM_CLASSES) X {obj_conf.N_LABEL} (SHOTS PER CLASS): {obj_conf.N_LABEL*len(classes)}")
        log.info(f"NUMBER OF SHOTS {len(few_shots_labs)}")
        
        # Define the set of unlabeled data which excludes the few samples labeled data
        unseen_labeled_files = []
        unseen_labeles = []
        for idx, f in enumerate(labeled_files):
            if f not in few_shots_files:
                unseen_labeled_files += [f]
                unseen_labeles += [labeles[idx]]

        log.info(f"Size of unlabeled data: {len(unseen_labeled_files)}")
        
        # Define the few shots as the labeled data
        labeled_files = few_shots_files
        labeles = few_shots_labs
    else:
        # Create datasets for TRZSL
        labeled_files, labeles = zip(*labeled_data)
        unseen_labeled_files, unseen_labeles = zip(*unlabeled_data)
        test_labeled_files, test_labeles = zip(*test_data)

    # Separate train and validation
    np.random.seed(obj_conf.validation_seed)
    train_indices = np.random.choice(
        range(len(labeled_files)),
        size=int(len(labeled_files) * obj_conf.ratio_train_val),
        replace=False,
    )
    val_indices = list(set(range(len(labeled_files))).difference(set(train_indices)))

    train_labeled_files = np.array(labeled_files)
    train_labeles = np.array(labeles)

    val_labeled_files = np.array(labeled_files)[val_indices]
    val_labeles = np.array(labeles)[val_indices]

    DatasetObject = dataset_object(obj_conf.DATASET_NAME)
    augmentation = (augmentations.build_transform("randomresizedcrop"), None)
    
    # Labeled training set
    train_seen_dataset = DatasetObject(
        train_labeled_files,
        data_folder,
        transform=None, # Set later 
        augmentations=augmentation,
        train=True,
        labels=train_labeles,
        label_map=label_to_idx,
    )
    # Unlabeled training set 
    train_unseen_dataset = DatasetObject(
        unseen_labeled_files,
        data_folder,
        transform=None,
        augmentations=augmentation,
        train=True,
        labels=unseen_labeles,
        label_map=label_to_idx,
    )
    main_samples = DatasetObject(
        [],
        data_folder,
        transform=None,
        augmentations=None,
        train=True,
        labels=[],
        label_map=label_to_idx,
    )

    # Adjust the name file to correctly load data
    truncated_unseen_labeled_files = [i.split("/")[-1] for i in train_unseen_dataset.filepaths]

    # Validation set (labeled data)
    val_seen_dataset = DatasetObject(
        val_labeled_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=True,
        labels=val_labeles,
        label_map=label_to_idx,
    )
    # Test set
    test_dataset = DatasetObject(
        test_labeled_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=False,
        labels=None,
        label_map=label_to_idx,
    )
    # Log info data
    log.info(f"\n----------------------TRAINING DATA INFO-----------------------\n")
    log.info(f"Size labeled data: {len(train_seen_dataset.filepaths)}")
    log.info(f"Size unlabeled data: {len(train_unseen_dataset.filepaths)}")
    log.info(f"Size validation data: {len(val_seen_dataset.filepaths)}")
    log.info(f"Size test data: {len(test_dataset.filepaths)}")
    log.info(f"\n-------------------------------------------------------------\n")
    # Define model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TextualFPL(
        obj_conf, 
        label_to_idx, 
        data_folder,
        dataset_obj=DatasetObject,
        unlabeled_files=truncated_unseen_labeled_files,
        device=device, 
        **dict_classes
    )
    test_acc = model.train(
        train_seen_dataset, 
        val_seen_dataset,
        main_samples=main_samples,
        unlabeled_data=train_unseen_dataset,
        test_data=test_dataset,
        only_seen=False,
        test_labeled_files=test_labeled_files,
        test_labeles=test_labeles
    )

    # # Store model results
    store_results(obj_conf, test_acc)

    # Validate on test set (standard)
    images, predictions, logits, text_features = model.evaluation(test_dataset)

    dictionary_predictions = {
        'images' : images, 
        'predictions' : predictions,
        'labels' : test_labeles,
        'text_features' : text_features,
        'logits' : logits,
    }

    if accelerator.is_local_main_process:
        save_predictions(dictionary_predictions, obj_conf, iteration=None)

 
def main():
    parser = argparse.ArgumentParser(description="Run task")
    parser.add_argument(
        "--model_config",
        type=str,
        default="model_config.yml",
        help="Name of model config file",
    )

    args = parser.parse_args()

    with open(f"methods_config/{args.model_config}", "r") as file:
        config = yaml.safe_load(file)

    # Cast configs to object
    obj_conf = Config(config)

    # Set seed
    optim_seed = int(os.environ["OPTIM_SEED"])
    obj_conf.OPTIM_SEED = optim_seed
    # Set backbone
    obj_conf.VIS_ENCODER = os.environ["VIS_ENCODER"]
    # Set dataset name
    obj_conf.DATASET_NAME = os.environ["DATASET_NAME"]
    if obj_conf.DATASET_NAME == 'Flowers102':
        obj_conf.N_MAIN_SAMPLES =  6
        # obj_conf.MAX_MAIN_RATIO = 0.5
    if obj_conf.DATASET_NAME == 'FGVCAircraft':
        obj_conf.THRESHOLD = 0.5
        obj_conf.MAX_MAIN_RATIO = 0.6
        obj_conf.MARGIN_SCALE = 14.0
    # Set dataset dir
    obj_conf.DATASET_DIR = os.environ["DATASET_DIR"]
    # Set split seed
    obj_conf.SPLIT_SEED = int(os.environ["SPLIT_SEED"])
    obj_conf.LR = float(os.environ["LR"])
    obj_conf.MARGIN_SCALE = float(os.environ["MARGIN_SCALE"])
    # Define dataset's template for textual prompts
    obj_conf.PROMPT_TEMPLATE = dataset_custom_prompts[obj_conf.DATASET_NAME]
    # Set data dir
    dataset_dir = obj_conf.DATASET_DIR
    
    # Set the file path for the log file
    log_file = f"logs/{obj_conf.DATASET_NAME}_{obj_conf.VIS_ENCODER.replace('/', '-')}_{os.environ.get('CUDA_VISIBLE_DEVICES')}.log"
    # Create a FileHandler and set the log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    # Add the FileHandler to the logger
    logger_.addHandler(file_handler)

    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Dataset dir: {dataset_dir}")

    # Check dataset directory exists
    if not Path(dataset_dir).exists():
        print(dataset_dir)
        raise Exception("`dataset_dir` does not exist..")

    # Set random seeds
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(obj_conf.OPTIM_SEED)
    random.seed(obj_conf.OPTIM_SEED)
    torch.manual_seed(obj_conf.OPTIM_SEED)
    accelerator.wait_for_everyone()
    # Seed for cuda
    if torch.cuda.is_available():
        torch.cuda.manual_seed(obj_conf.OPTIM_SEED)
        torch.cuda.manual_seed_all(obj_conf.OPTIM_SEED)
        accelerator.wait_for_everyone()

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    workflow(dataset_dir, obj_conf)


if __name__ == "__main__":
    main()
