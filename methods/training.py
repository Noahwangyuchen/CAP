import copy
import datetime
from functools import reduce
import logging
import math
from operator import mul
from models.maple import CustomCLIP, load_clip_to_cpu
from utils.compute_metrics import evaluate_predictions
from utils.data import ForeverDataIterator

import clip
import numpy as np
import pandas as pd
import scipy.stats as st
import torch
import torchvision.transforms as transforms
from accelerate import Accelerator
from PIL import Image
from torch import nn
from torch.nn.modules.utils import _pair
import torch.utils

accelerator = Accelerator()

from models import (
    CustomImageEncoder, 
    CustomTextEncoder, 
    ImagePrefixModel,
    TextPrefixModel,
    DebiasModule,
    UPTModel,
    UPT_Adapter,
    CrossAdapter,
)
from utils import (
    make_scheduler, 
    seed_worker, 
    save_parameters,
    save_pseudo_labels,
)


g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)

class TrainingStrategy(object):
    def __init__(
        self, 
        config, 
        label_to_idx, 
        classes, 
        seen_classes, 
        unseen_classes, 
        device
    ):
        """ This class defines functions for the training strategies.

        :param config: dictionaries of prameters in models_config/vpt_baseline_config.yml
        :param label_to_idx: dictionary (key, value):(class name, id)
        :param classes: list of class names
        :param seen_classes: list of seen classes' names
        :param unseen_classes: list of unseen classes' names
        :param device: device in use
        """

        self.config = config
        self.classes = classes
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.label_to_idx = label_to_idx

        self.device = device
        self.template = self.config.PROMPT_TEMPLATE
        self.clip_model, self.transform = clip.load(
            self.config.VIS_ENCODER, device=self.device
        )

    def define_model(self, classes=None):
        """ This function initialized the model
        depending on the prompt modality.

        :param modality: either text or image
        :param classes: the list of classes for textual model
        """

        cfg = self.config
        classes = self.classes
        clip_model = load_clip_to_cpu(cfg)

        if cfg.PREC == "fp32" or cfg.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        
        self.model = CustomCLIP(cfg, classes, clip_model).to(self.device)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)


        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        for i, parameter in enumerate(self.model.parameters()):
            if parameter.requires_grad:
                log.info(f"Shape of parameters {i}: {parameter.shape}")

        self.dim = 768 if self.config.VIS_ENCODER == "ViT-L/14" else 512
        self.debias_module = DebiasModule(in_dim=self.dim, device=self.device, dtype=self.dtype, cfg=self.config).to(self.device)

        if self.config.OPTIM == "SGD":
            self.prompt_optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.LR,
                weight_decay=self.config.DECAY,
                momentum=0.9,
            )
            self.debias_adapter_optimizer = torch.optim.SGD(
                self.debias_module.parameters(),
                lr=self.config.LR,
                weight_decay=self.config.DECAY,
                momentum=0.9,
            )

        self.prompt_scheduler = make_scheduler(self.prompt_optimizer, self.config)
        self.debias_scheduler = make_scheduler(self.debias_adapter_optimizer, self.config)
        self.loss_func = torch.nn.CrossEntropyLoss()
    
    def create_training_dataset(self, train_data, unlabeled_data=None, main_samples=None):
        """This function create the dataset for training. Specifically, it
        merges pseudo-labels for unseen data and labeled data for seen classes.

        :param train_data: Dataset object - training seen classes (defined in zsl_jpl line 323)
        :param unlabeled_data: Dataset object - dataset of unlabeled data for
                               unseen classes (defined in zsl_jpl line 328)
        """
        self.val_unseen_files = None
        return train_data
    
    def train(
        self,
        train_data,
        val_data,
        main_samples=None,
        unlabeled_data=None,
        only_unlabelled=False,
        only_seen=False,
        iterative=False,
        test_data=None,
        test_labeled_files=None,
        test_labeles=None,
    ):
        """This function defines the training of self.model.

        :param train_data: Dataset object - training dataset of labeled data for
                           seen classes (defined in zsl_jpl line 323)
        :param val_data: Dataset object - validation dataset of labeled data for
                         seen classes (defined in zsl_jpl line 334)
        :param unlabeled_data: Dataset object - dataset of unlabeled data for
                               unseen classes (defined in zsl_jpl line 328)
        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        """
        
        # Define training dataset
        if not iterative:
            self.create_training_dataset(train_data, unlabeled_data, main_samples)
            self.unlabeled_samples_per_class = len(unlabeled_data.filepaths) // len(self.unseen_classes)
            
            if accelerator.is_local_main_process:
                for class_idx, samples in self.leaderboard.items():
                    samples = [(confidence.item(), filepath.split("/")[-1]) for confidence, filepath in samples]
                    log.info(f"Class {self.classes[class_idx]}: {samples}")
        
            self.define_model(self.classes)

        log.info(f"[self.train] Training data: {len(train_data.filepaths)}")
        
        # Declare the data pre processing for train and validation data
        train_data.transform = self.transform
        unlabeled_data.transform = self.transform
        main_samples.transform = self.transform
        val_data.transform = self.transform

        if self.config.PARADIGM != 'UL':
            labeled_data_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g,
            )
        else:
            labeled_data_loader = None

        main_samples_loader = torch.utils.data.DataLoader(
            main_samples,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        unlabeled_data_loader = torch.utils.data.DataLoader(
            unlabeled_data,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=True,
        )

        main_samples_loader = ForeverDataIterator(main_samples_loader)
        unlabeled_data_loader = ForeverDataIterator(unlabeled_data_loader)
        if labeled_data_loader is not None:
            labeled_data_loader = ForeverDataIterator(labeled_data_loader)

        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=self.config.BATCH_SIZE
        )

        accelerator.wait_for_everyone()

        self.model, self.debias_module, self.prompt_optimizer, self.debias_adapter_optimizer, main_samples_loader, unlabeled_data_loader, val_loader = accelerator.prepare(
            self.model, self.debias_module, self.prompt_optimizer, self.debias_adapter_optimizer, main_samples_loader, unlabeled_data_loader, val_loader
        )
        if labeled_data_loader is not None:
            labeled_data_loader = accelerator.prepare(labeled_data_loader)

        best_val_accuracy = 0
        best_prompt = None
        test_acc = 0.0
        loss = None
        if val_loader is not None and accelerator.is_local_main_process:
            log.info(f"Size of validation dataset: {len(val_data.filepaths)}")

        for epoch in range(self.config.EPOCHS):
            if accelerator.is_local_main_process:
                log.info(f"Run Epoch {epoch}")

            total_loss = 0
            accum_iter = self.config.ACCUMULATION_ITER

            loss, total_loss, new_leader_board = self._train_epoch(
                loss,
                total_loss,
                accum_iter,
                epoch,
                main_samples_loader,
                labeled_data_loader,
                unlabeled_data_loader,
                only_unlabelled=only_unlabelled,
                only_seen=only_seen,
            )
            accelerator.wait_for_everyone()

            if epoch % self.config.ITER_EPOCHS == 0:
                main_samples_loader = self.update_main_samples(new_leader_board, main_samples, main_samples_loader, epoch)

                # self.prompt_scheduler, self.debias_scheduler = accelerator.prepare(self.prompt_optimizer, self.debias_scheduler)

            # accelerator.free_memory()

            # if val_loader is not None:
            #     val_accuracy = self._run_validation(val_loader, only_unlabelled)
            #     if accelerator.is_local_main_process:
            #         log.info(f"Validation accuracy after Epoch {epoch}: {val_accuracy}")
            # else:
            #     best_val_accuracy = None

            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.config.EPOCHS:
                std_predictions = self.test_predictions(test_data)
                test_acc = evaluate_predictions(
                    self.config,
                    std_predictions,
                    test_labeled_files,
                    test_labeles,
                    self.unseen_classes,
                    self.seen_classes
                )
                if accelerator.is_local_main_process:
                    log.info(f"ZSL accuracy: {test_acc}")
                self.debias_module.module.cfg.INFERENCE_MODE = 'direct'
                std_predictions = self.test_predictions(test_data)
                test_acc = evaluate_predictions(
                    self.config,
                    std_predictions,
                    test_labeled_files,
                    test_labeles,
                    self.unseen_classes,
                    self.seen_classes
                )
                self.debias_module.module.cfg.INFERENCE_MODE = 'main'
                if accelerator.is_local_main_process:
                    log.info(f"ZSL accuracy: {test_acc}")
            
            accelerator.wait_for_everyone()
        
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saved_models/model——{current_time}.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'debias_module_state_dict': self.debias_module.state_dict(),
        }, filename)

        return test_acc
    
    def update_main_samples(self, new_leaderboard, main_samples, main_samples_loader, epoch):
        """This function updates the main samples for the next epoch.

        :param new_leaderboard: dictionary (key, value):(class id, list of (confidence, filepath))
        :param main_samples: Dataset object
        :param main_samples_loader: DataLoader object
        """

        # TODO: 不同的样本替换方案
        updated_leaderboard = {self.label_to_idx[self.unseen_classes[i]]: [] for i in range(len(self.unseen_classes))}
        k = self.config.N_MAIN_SAMPLES + int((min(1000, int(self.unlabeled_samples_per_class * self.config.MAX_MAIN_RATIO)) - self.config.N_MAIN_SAMPLES) * min(1, epoch / (self.config.EPOCHS * 0.7)))
    
        for classname in self.unseen_classes:
            samples = []
            class_id = self.label_to_idx[classname]
            samples.extend(new_leaderboard[class_id][:k])
            # for sample in self.leaderboard[class_id]:
            #     if sample[1] not in [s[1] for s in samples]:
            #         samples.append(sample)
            samples.extend(self.leaderboard[class_id][:k-len(samples)])
            updated_leaderboard[class_id] = samples

        for classname in self.unseen_classes:
            self.leaderboard[self.label_to_idx[classname]] = updated_leaderboard[self.label_to_idx[classname]]

        new_imgs = []
        new_labels = []
        for idx, leaderboard in updated_leaderboard.items():
            new_imgs += [sample[1] for sample in leaderboard]
            new_labels += [idx] * len(leaderboard)

        main_samples.filepaths = new_imgs
        main_samples.labels = new_labels

        main_samples_loader = torch.utils.data.DataLoader(
            main_samples,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        main_samples_loader = ForeverDataIterator(main_samples_loader)
        main_samples_loader = accelerator.prepare(main_samples_loader)

        if accelerator.is_local_main_process:
            for class_id, samples in self.leaderboard.items():
                if self.classes[class_id] in self.unseen_classes:
                    samples_for_log = [(confidence, path.split("/")[-1]) for confidence, path in samples]
                    log.info(f"leaderboard for class {self.classes[class_id]}: {samples_for_log}")

        return main_samples_loader

    def define_loss_function(self, logits, labs, paths):
        return self.loss_func(logits, labs)

    def backpropagate(self):
        self.prompt_optimizer.step()
        self.debias_adapter_optimizer.step()
        self.clear_grad()

    def clear_grad(self):
        self.prompt_optimizer.zero_grad()
        self.debias_adapter_optimizer.zero_grad()
        self.model.zero_grad()

    def update_scheduler(self):
        self.prompt_scheduler.step()
        self.debias_scheduler.step()

    def unwrap_model(self):
        return accelerator.unwrap_model(self.model)

    def training_model(self, img):
        """This function allows to customize the model to use while trainig

        :param img: Tensor of images form Dataloader
        """
        return self.model(img)
