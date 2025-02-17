import logging

import custom_clip
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from PIL import Image
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

accelerator = Accelerator()

from methods import TextualPrompt
from utils import (
    dataset_object,
    make_scheduler, 
    pseudolabel_top_k, 
    seed_worker,
)


g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)


class TextualFPL(TextualPrompt):
    def __init__(
        self,
        config,
        label_to_idx,
        data_folder,
        unlabeled_files,
        classes,
        seen_classes,
        unseen_classes,
        device,
        dataset_obj=None,
    ):
        """This class define Coop baseline.

        :param config: dictionaries of prameters in models_config/coop_baseline_config.yml
        :param label_to_idx: dictionary (key, value):(class name, id)
        :param classes: list of class names
        :param seen_classes: list of seen classes' names
        :param unseen_classes: list of unseen classes' names
        :param device: device in use
        """
        super().__init__(
            config, label_to_idx, classes, seen_classes, unseen_classes, device
        )

        self.data_folder = data_folder

        self.check_unlabeled = unlabeled_files

        self.dataset_obj = dataset_obj


    def create_training_dataset(self, train_data, unlabeled_data=None, main_samples=None):
        """This function create the dataset for training. Specifically, it
        merges pseudo-labels for unseen data and labeled data for seen classes.

        :param train_data: Dataset object - training seen classes (defined in zsl_jpl line 323)
        :param unlabeled_data: Dataset object - dataset of unlabeled data for
                               unseen classes (defined in zsl_jpl line 328)
        """

        unseen_imgs = unlabeled_data.filepaths
        unseen_labs = unlabeled_data.labels
        
        self.leaderboard = pseudolabel_top_k(
            self.config,
            self.config.DATASET_NAME,
            self.config.N_MAIN_SAMPLES,
            self.config.PROMPT_TEMPLATE,
            train_data.filepaths,
            train_data.labels,
            unlabeled_data.filepaths,
            self.unseen_classes,
            self.transform,
            self.clip_model,
            self.label_to_idx,
            self.device,
            self.config.VIS_ENCODER,
            self.config.SPLIT_SEED
        )
        
        self.seen_imgs = train_data.filepaths
        self.seen_labs = [self.label_to_idx[l] for l in train_data.labels]
        
        # In TRZSL, we need to obtain leaderboard for seen classes to compute the similarity aware margin.
        if self.config.PARADIGM == 'TRZSL':
            for class_name in self.seen_classes:
                self.leaderboard[self.label_to_idx[class_name]] = []
            for i in range(len(self.seen_imgs)):
                lab = self.seen_labs[i]
                if self.idx_to_real[lab] in self.seen_classes and len(self.leaderboard[lab]) < self.config.N_MAIN_SAMPLES:
                    self.leaderboard[lab].append((torch.tensor(1.0, dtype=self.dtype, device=self.device), self.seen_imgs[i]))

        pseudo_imgs = []
        pseudo_labs = []
        for index, leaderboard in self.leaderboard.items():
            if self.config.PARADIGM == 'TRZSL' and self.idx_to_real[index] in self.seen_classes:
                continue
            pseudo_imgs += [tup[1] for tup in leaderboard]
            pseudo_labs += [index] * len(leaderboard)

        self.val_unseen_files = None
        self.val_unseen_labs = None

        unseen_labs = [self.label_to_idx[l] for l in unseen_labs]

        unlabeled_data.filepaths = unseen_imgs
        unlabeled_data.labels = unseen_labs
        unlabeled_data.label_id = True

        train_data.filepaths = list(self.seen_imgs)
        train_data.labels = list(self.seen_labs)
        train_data.label_id = True

        main_samples.filepaths = pseudo_imgs
        main_samples.labels = pseudo_labs
        main_samples.label_id = True

    def define_loss_function(self, logits, labs, paths):

        loss_ce_seen = self.cross_entropy(logits, labs, paths, False)
        loss_ce_unseen = self.cross_entropy(logits, labs, paths, True)

        return self.balance_param * loss_ce_seen + loss_ce_unseen

    def cross_entropy(self, logits, labels, paths, unlabeled=True):
        """This loss computes the probability mass on the1
        opposite set of classes for each sample.

        :param logits: continuous vector
        :param labels: class ids
        """

        # self.check_unlabeled
        if unlabeled:
            samples = []
            for idx in range(len(paths)):
                if paths[idx] in self.check_unlabeled:
                    samples.append(idx)

            # log.info(f"Unlabeled: {len(samples)} {self.balance_param}")
            if samples:
                error = self.loss_func(logits[samples], labels[samples])
            else:
                error = 0
        else:
            samples = []
            for idx in range(len(paths)):
                if paths[idx] not in self.check_unlabeled:
                    samples.append(idx)

            # log.info(f"Labeled: {len(samples)} {self.balance_param}")
            if samples:
                error = self.loss_func(logits[samples], labels[samples])
            else:
                error = 0
        
        return error

    def shift_log(self, x, offset=1e-5):
        return torch.log(torch.clamp(x + offset, max=1.))
    
    def worst_case_estimation_loss(self, y_l, y_l_adv, y_u, y_u_adv, mask):
        """
        Worst-case Estimation loss from `Debiased Self-Training for Semi-Supervised Learning`.
        Forces the worst-case head h_worst to predict correctly on all labeled samples
        while making as many mistakes as possible on unlabeled data.

        Args:
            eta_prime (float): the trade-off hyperparameter η'.
            y_l (tensor): logits output by the main head on labeled data.
            y_l_adv (tensor): logits output by the worst-case estimation head on labeled data.
            y_u (tensor): logits output by the main head on unlabeled data.
            y_u_adv (tensor): logits output by the worst-case estimation head on unlabeled data.

        Returns:
            (tensor): The computed loss value.
        """

        # Loss on labeled data
        _, prediction_l = y_l.max(dim=1)
        loss_l = self.eta_prime * F.cross_entropy(y_l_adv, prediction_l)

        # Loss on unlabeled data
        _, prediction_u = y_u.max(dim=1)
        loss_u = (F.nll_loss(self.shift_log(1. - F.softmax(y_u_adv, dim=1)), prediction_u, reduction='none') * mask).sum()
        n_unlabeled = mask.sum()
        if n_unlabeled > 0:
            loss_u = loss_u / n_unlabeled

        return loss_l, loss_u
    
    def balanced_cross_entropy_loss(self, logits, labels, paths):
        """Cross entropy loss with balance factor for labeled data and pseudo-labeled main samples."""

        weights = torch.ones_like(labels, dtype=self.dtype, device=self.device)
        for idx, path in enumerate(paths):
            if path in self.seen_imgs:
                weights[idx] = self.balance_factor
        
        loss = F.cross_entropy(logits, labels, reduction='none')
        loss = (loss * weights).mean()
        
        return loss
    
    def confidence_based_self_training_loss(self, y, pseudo_labels, mask):
        self_training_loss = (F.cross_entropy(y, pseudo_labels, reduction='none') * mask).sum()
        n_pseudo_labels = mask.sum()
        if n_pseudo_labels > 0:
            self_training_loss = self_training_loss / n_pseudo_labels

        return self_training_loss

    def entropy(self, x):
        epsilon = 1e-5
        entropy = -x * torch.log(x + epsilon)
        return entropy.sum(dim=1)