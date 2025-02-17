import logging

import custom_clip
import numpy as np
import pandas as pd
import copy
import scipy.stats as st
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from torch import nn

accelerator = Accelerator()

from models import CustomTextEncoder, TextPrefixModel
from methods import TrainingStrategy
from utils import make_scheduler, seed_worker

g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)


class TextualPrompt(TrainingStrategy):
    def __init__(
        self,
        config,
        label_to_idx,
        classes,
        seen_classes,
        unseen_classes,
        device,
    ):
        """This class define Coop's training and evaluation.

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

        # Build dictionaries to correctly label model's predictions
        self.idx_to_real = {
            label_to_idx[c]: c for c in self.classes
        }
        
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.margin_scale = self.config.MARGIN_SCALE
        num_classes = len(self.classes)
        self.margin_mask = torch.ones((num_classes, num_classes)).to(self.device)
        self.margin_mask[torch.arange(num_classes), torch.arange(num_classes)] = 0.0

        self.threshold = config.THRESHOLD
        if self.config.N_LABEL != 0:
            self.balance_factor = float(config.N_MAIN_SAMPLES / config.N_LABEL)
        else:
            self.balance_factor = 1.0

        self.pl_per_class = {}

    def _train_epoch(
        self, 
        loss, 
        total_loss, 
        accum_iter, 
        epoch, 
        main_sample_loader, 
        labeled_data_loader,
        unlabeled_data_loader,
        only_unlabelled=False,
        only_seen=False,
    ):
        """This function defines the training epoch of self.model.

        :param loss: float loss (average across batches)
        :param total_loss: float total loss
        :param train_loader: Dataloader object - training data defined in self.train
        :param accum_iter: number of accumulation steps minimum 1
        :param epoch: current epoch
        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        :param only_seen: boolean.  It is True if the training only involves
                                seen data

        """
        margin = self.compute_margin()
        
        
        predictions = []

        all_pseudo_labels = []
        self.pl_per_class = {}

        for i in range(self.config.ITER_PER_EPOCH):
            (x_m, _, _, m_label, m_path), (x_u, x_u_aug, _, u_label, u_path) = next(main_sample_loader), next(unlabeled_data_loader)

            if labeled_data_loader is not None:
                (x_l, _, _, l_label, l_path) = next(labeled_data_loader)
                x_l, l_label = x_l.to(self.device), l_label.to(self.device)

            x_m, x_u, x_u_aug, m_label, u_label = x_m.to(self.device), x_u.to(self.device), x_u_aug.to(self.device), m_label.to(self.device), u_label.to(self.device)

            self.clear_grad()

            # extract features
            t_m, v_m, _ = self.extract_features(x_m)
            t_u, v_u, _ = self.extract_features(x_u)
            t_u_aug_ps, _, v_u_aug_ps = self.extract_features(x_u_aug)
            if labeled_data_loader is not None:
                t_l, v_l, _ = self.extract_features(x_l)

            # cosine similarity as logits
            logit_scale = self.model.module.logit_scale.exp()
            main_sample_logits = logit_scale * v_m @ t_m.t()
            unlabeled_logits = logit_scale * v_u @ t_u.t()
            unlabeled_logits_ps = logit_scale * v_u_aug_ps @ t_u_aug_ps.t()
            if labeled_data_loader is not None:
                labeled_logits = logit_scale * v_l @ t_l.t()

            loss = torch.tensor(0.0).to(self.device)

            # Cross-entropy loss for labeled data
            if labeled_data_loader is not None:
                labs = self.get_labels_id(only_seen, l_label)
                margin_labeled_logits = labeled_logits + margin[labs] * self.margin_mask[labs]
                labeled_loss = F.cross_entropy(margin_labeled_logits, l_label)
                loss += labeled_loss

            # Cross-entropy loss for main samples
            labs = self.get_labels_id(only_seen, m_label)
            marigined_main_sample_logits = main_sample_logits + margin[labs] * self.margin_mask[labs]
            main_sample_loss = F.cross_entropy(marigined_main_sample_logits, m_label)
            loss += main_sample_loss

            # Self-training loss for unlabeled data
            confidence, pseudo_labels = F.softmax(unlabeled_logits.detach(), dim=1).max(dim=1)
            mask = (confidence > self.threshold).float()
            margined_unlabeled_logits_ps = unlabeled_logits_ps + margin[pseudo_labels] * self.margin_mask[pseudo_labels]
            self_training_loss = self.confidence_based_self_training_loss(margined_unlabeled_logits_ps, pseudo_labels, mask)
            
            loss += self_training_loss

            # collect confidence score and pseudo labels for unlabeled data
            pseudo_labels = pseudo_labels * mask - (1 - mask)
            # TODO: 可以用 predictions 中的数据删除 all_pseudo_labels
            all_pseudo_labels += [i for i in pseudo_labels]
            # TODO: 尝试加 mask 或者不加 mask
            for j in range(x_u.size(0)):
                if mask[j] > 0:
                    predictions.append((pseudo_labels[j].item(), confidence[j].item(), u_path[j]))
            
            # pseudo label accuracy
            n_pseudo_labels = mask.sum()
            pseudo_label_acc = 0.0
            u_labs = self.get_labels_id(only_seen, u_label)
            if n_pseudo_labels > 0:
                n_correct = (pseudo_labels == u_labs).float().sum()
                pseudo_label_acc = n_correct / n_pseudo_labels * 100

            accelerator.wait_for_everyone()

            loss = loss / accum_iter
            accelerator.backward(loss)
            self.backpropagate()
            accelerator.wait_for_everyone()
            self.update_scheduler()

            # loggings
            if accelerator.is_local_main_process:
                if labeled_data_loader is not None:
                    log.info(f"labeled_loss: {labeled_loss.item()}")
                log.info(f"main_samples_loss: {main_sample_loss.item()}")
                log.info(f"n_pseudo_labels: {n_pseudo_labels}, self_training_loss: {self_training_loss.item()}")
                log.info(f"Pseudo label accuracy: {pseudo_label_acc}%")
                # log.info(f"mcc_loss: {mcc_loss.item()}")
                log.info(f"lr of prompt_scheduler: {self.prompt_scheduler.get_last_lr()}")
                log.info(f"lr of debias_scheduler: {self.debias_scheduler.get_last_lr()}")

        accelerator.wait_for_everyone()
        all_pseudo_labels = torch.tensor(all_pseudo_labels).to(self.device)
        all_pseudo_labels = accelerator.gather(all_pseudo_labels)
        predictions = accelerator.gather_for_metrics(predictions)

        # compute pseudo label distribution
        for i in range(len(self.classes)):
            tot_pl = (all_pseudo_labels == i).sum().item()
            self.pl_per_class[self.classes[i]] = tot_pl

        # compute top-k confident samples for each class
        top_k_leaderboard = self.compute_top_k_leaderboard(predictions, main_sample_loader.data_loader.dataset.filepaths)

        if accelerator.is_local_main_process:
            log.info(f"Pseudo label generating per class: {self.pl_per_class}")

        return loss, total_loss, top_k_leaderboard

    def compute_top_k_leaderboard(self, predictions, main_samples_paths):
        """This function computes the top-k confident samples for each class, k = main_samples - labeled_samples.

        :param predictions: list of tuples (pseudo_label, confidence, img_path)
        :param main_samples: current filepaths of main samples
        """

        leaderboard = {self.label_to_idx[self.unseen_classes[i]]: [] for i in range(len(self.unseen_classes))}
        k = self.config.N_MAIN_SAMPLES
        seen_img_paths = set(main_samples_paths)

        for pseudo_label, confidence, img_path in predictions:
            if img_path in seen_img_paths or pseudo_label not in leaderboard.keys():
            # if pseudo_label not in leaderboard.keys():
                continue
            leaderboard[pseudo_label].append((confidence, img_path))
        
        # select top-k samples for each class
        selected_img_paths = set()
        top_k_leaderboard = {self.label_to_idx[self.unseen_classes[i]]: [] for i in range(len(self.unseen_classes))}
        for class_id, samples in leaderboard.items():
            if len(samples) == 0:
                continue
            samples = sorted(samples, reverse=True)
            for confidence, img_path in samples:
                if img_path not in selected_img_paths:
                    top_k_leaderboard[class_id].append((confidence, img_path))
                    selected_img_paths.add(img_path)
                    if len(top_k_leaderboard[class_id]) >= 1000:
                        break
        # if accelerator.is_local_main_process:
        #     for class_id, samples in top_k_leaderboard.items():
        #         samples_for_log = [(confidence, path.split("/")[-1]) for confidence, path in samples]
        #         log.info(f"leaderboard for class {self.classes[class_id]}: {samples_for_log}")
            

        return top_k_leaderboard


    def compute_margin(self):
        """This function computes the margin scale for each class based on the number of pseudo-labeled samples
        and similarities between embeddings.

        :param main_sample_loader: Dataloader object - main_samples
        """

        if len(self.pl_per_class) > 0:
            # compute similarity between text embeddings
            with torch.no_grad():
                text_features, _ = self.model(None)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features, _ = self.debias_module(text_out=text_features, inference=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                textual_similarity = text_features @ text_features.t()

            # compute similarity between visual embeddings
            main_imgs = []
            main_labels = []
            for idx, leaderboard in self.leaderboard.items():
                main_imgs += [sample[1] for sample in leaderboard]
                main_labels += [idx] * len(leaderboard)

            tmp_dataset = self.dataset_obj(
                main_imgs,
                self.data_folder,
                transform=self.transform,
                augmentations=None,
                train=False,
                labels=main_labels,
                label_map=self.label_to_idx,
            )
            tmp_dataset.filepaths = main_imgs
            tmp_dataset.labels = main_labels
            tmp_dataset.label_id = True
            tmp_data_loader = torch.utils.data.DataLoader(
                tmp_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g,
            )
            
            tmp_data_loader = accelerator.prepare(tmp_data_loader)
            visual_embeddings = []
            for x_l, _, _, label, filepath in tmp_data_loader:
                with torch.no_grad():
                    x_l = x_l.to(self.device)
                    label = label.to(self.device)
                    _, v_l, _ = self.extract_features(x_l)
                    v_l = v_l / v_l.norm(dim=-1, keepdim=True)
                    for j in range(v_l.size(0)):
                        visual_embeddings.append((label[j].item(), v_l[j], filepath[j]))
            accelerator.wait_for_everyone()
            visual_embeddings = accelerator.gather_for_metrics(visual_embeddings)
            class_v_sum = torch.zeros((len(self.classes), self.dim), dtype=self.dtype).to(self.device)
            class_v_mean = torch.zeros((len(self.classes), self.dim), dtype=self.dtype).to(self.device)
            file_paths = set()
            for label, v, file_path in visual_embeddings:
                if file_path not in file_paths:
                    class_v_sum[label] += v.to(self.device)
                    file_paths.add(file_path)
            for class_name in self.classes:
                if len(self.leaderboard[self.label_to_idx[class_name]]) > 0:
                    class_v_mean[self.label_to_idx[class_name]] = class_v_sum[self.label_to_idx[class_name]] / len(self.leaderboard[self.label_to_idx[class_name]])
                else:   # no pseudo-labeled samples for this class, use text feature
                    class_v_mean[self.label_to_idx[class_name]] = text_features[self.label_to_idx[class_name]]
            
            class_v_mean = class_v_mean / class_v_mean.norm(dim=-1, keepdim=True)
            visual_similarity = class_v_mean @ class_v_mean.t()

            # TODO: 不同的fusion方式
            # combined_similarity = (textual_similarity + visual_similarity) / 2.0
            combined_similarity = torch.max(textual_similarity, visual_similarity)
            x_min, _ = torch.min(combined_similarity, dim=1, keepdim=True)
            x_max, _ = torch.max(combined_similarity, dim=1, keepdim=True)
            combined_similarity = x_min + ((combined_similarity - x_min) / (x_max - x_min)) * (1 - x_min)
            
            class_ratio = torch.zeros((len(self.classes),)).to(self.device)
            max_val = max(self.pl_per_class.values())
            for key, value in self.pl_per_class.items():
                if key in self.unseen_classes:
                    class_ratio[self.label_to_idx[key]] = value / max_val
                else:
                    class_ratio[self.label_to_idx[key]] = 1.0
            mar_diff = (class_ratio.max() - class_ratio.min()) * self.margin_scale
            # mar_diff = 0.3 * self.margin_scale
            margin = (((1.0 - class_ratio) / (1.0 + class_ratio)) * mar_diff).unsqueeze(1) * combined_similarity
            # margin = mar_diff * combined_similarity

        else:
            margin = torch.zeros((len(self.classes), len(self.classes))).to(self.device)

        return margin

    def get_labels_id(self, only_seen, all_label):
        if only_seen:
            labs = torch.tensor([self.real_to_idx[l.item()] for l in all_label]).to(
                    self.device
                )
        else:
            labs = torch.tensor([l.item() for l in all_label]).to(self.device)
        return labs

    def extract_features(self, img):
        if torch.cuda.is_available():
            text_features, image_features = self.model(img)
            if torch.isnan(text_features).any() or torch.isnan(image_features).any():
                log.info(f"NAN DETECTED here")
                log.info(f"text_features: {text_features}")
                log.info(f"image_features: {image_features}")
                exit()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if torch.isnan(text_features).any() or torch.isnan(image_features).any():
            log.info(f"NAN DETECTED in t or v")
            log.info(f"text_features: {text_features}")
            log.info(f"image_features: {image_features}")
            exit()
        
        # t, v, t_adv, v_adv, t_ps, v_ps = self.debias_module(text_features, image_features)
        # t, v, t_adv, v_adv, t_ps, v_ps = [tensor / tensor.norm(dim=-1, keepdim=True) for tensor in [t, v, t_adv, v_adv, t_ps, v_ps]]

        t, v, v_ps = self.debias_module(text_features, image_features)
        t, v, v_ps = [tensor / tensor.norm(dim=-1, keepdim=True) for tensor in [t, v, v_ps]]

        # if torch.isnan(t).any() or torch.isnan(v).any() or torch.isnan(t_adv).any() or torch.isnan(v_adv).any():
        #     log.info(f"NAN DETECTED IN FEATURES")
        #     log.info(f"t: {t}")
        #     log.info(f"v: {v}")
        #     log.info(f"t_adv: {t_adv}") 
        #     log.info(f"v_adv: {v_adv}")
        #     exit()

        # return t, v, t_adv, v_adv, t_ps, v_ps
        return t, v, v_ps

    def _run_validation(
        self, 
        val_loader,
        only_unlabelled=False, 
        only_seen=False,
    ):
        """This function computes the validation accuracy on labeled seen data.

        :param val_loder: Dataloader object - validation dataset
        """

        if accelerator.is_local_main_process:
            if torch.cuda.is_available():
                log.info(f"[self._run_validation] Number of prompts: {len(self.classes)}")
            else:
                log.info(f"[self._run_validation] Number of prompts: {len(self.model.classes)}")
        
        predictions = []
        labels = []
        for img, _, _, label, img_path in val_loader:
            with torch.no_grad():
                text_features, image_features, _ = self.extract_features(img)

            logit_scale = self.model.module.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            idx_preds = torch.argmax(logits, dim=1)
            if self.val_unseen_files is not None:
                real_preds = [self.classes[i.item()] for i in idx_preds]
            else:
                real_preds = [self.seen_classes[i.item()] for i in idx_preds]

            predictions += real_preds
            labels += [self.classes[i.item()] for i in label]

        accelerator.wait_for_everyone()

        predictions = torch.tensor(
            [self.label_to_idx[p] for p in predictions][: len(val_loader.dataset)]
        ).to(self.device)
        labels = torch.tensor(
            [self.label_to_idx[l] for l in labels][: len(val_loader.dataset)]
        ).to(self.device)

        predictions_outputs = accelerator.gather(predictions)
        labels_outputs = accelerator.gather(labels)

        log.info(f"[self._run_validation] shape text_features {len(text_features)}")
        
        accuracy = torch.sum(predictions_outputs == labels_outputs) / len(
                predictions_outputs
            )
        log.info(f"Validation accuracy after Epoch: {accuracy}")

        return accuracy

    def test_predictions(self, data):
        """This function computes predictions on test data.

        :param data: Dataset object - test dataset
        """

        # Declare the data pre processing
        data.transform = self.transform
        # Define the data loader
        test_loader = torch.utils.data.DataLoader(
            data, batch_size=self.config.BATCH_SIZE
        )

        test_loader = accelerator.prepare(test_loader)

        log.info(f"[self.test_predictions] Number of prompts: {len(self.classes)}")
        accelerator.wait_for_everyone()

        # Get prompts
        with torch.no_grad():
            text_features, _ = self.model(None)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features, _ = self.debias_module(text_out=text_features, inference=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        log.info(f"TEXT FEATURES SHAPE: {text_features.size()}")

        log.info(f"Start inference for test data")
        # This is required for distributed training
        test_files = test_loader.dataset.filepaths

        predictions = []
        images = []
        for img, _, _, img_path in test_loader:
            with torch.no_grad():
                _, image_features = self.model(img)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                _, image_features = self.debias_module(visual_out=image_features, inference=True)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
            # cosine similarity as logits
            logit_scale = self.model.module.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            idx_preds = torch.argmax(logits, dim=1)

            predictions += [self.classes[i] for i in idx_preds]

            images += [i for i in img_path]

        predictions = torch.tensor([self.label_to_idx[p] for p in predictions]).to(
            self.device
        )
        images = torch.tensor([test_files.index(img) for img in images]).to(self.device)

        accelerator.wait_for_everyone()

        predictions_outputs = accelerator.gather(predictions)
        image_outputs = accelerator.gather(images)

        predictions_outputs = [self.classes[p] for p in predictions_outputs]
        image_outputs = [test_files[i] for i in image_outputs]

        df_predictions = pd.DataFrame(
            {"id": image_outputs, "class": predictions_outputs}
        )
        df_predictions.drop_duplicates(subset=["id", "class"], inplace=True)

        return df_predictions
    
    def evaluation(self, data):
        """This function computes predictions on test data.

        :param data: Dataset object - test dataset
        """

        # Declare the data pre processing
        data.transform = self.transform
        # Define the data loader
        test_loader = torch.utils.data.DataLoader(
            data, batch_size=self.config.BATCH_SIZE
        )

        test_loader = accelerator.prepare(test_loader)

        log.info(f"[self.test_predictions] Number of prompts: {len(self.classes)}")
        accelerator.wait_for_everyone()

        # Get prompts
        with torch.no_grad():
            text_features, _ = self.model(None)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features, _ = self.debias_module(text_out=text_features, inference=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        log.info(f"TEXT FEATURES SHAPE: {text_features.size()}")

        log.info(f"Start inference for test data")
        # This is required for distributed training
        test_files = test_loader.dataset.filepaths
        file_to_index = {file_name: idx for idx, file_name in enumerate(test_files)}

        predictions = []
        images = []
        all_logits = []
        for img, _, _, img_path in test_loader:
            with torch.no_grad():
                _, image_features = self.model(img)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                _, image_features = self.debias_module(visual_out=image_features, inference=True)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
            # cosine similarity as logits
            logit_scale = self.model.module.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            idx_preds = torch.argmax(logits, dim=1)

            predictions += [self.classes[i] for i in idx_preds]
            images += [i for i in img_path]
            all_logits += logits

        # accelerator.wait_for_everyone()
        predictions_outputs = accelerator.gather_for_metrics(predictions)
        image_outputs = accelerator.gather_for_metrics(images)
        all_logits = accelerator.gather_for_metrics(all_logits, use_gather_object=True)

        df_predictions = pd.DataFrame(
            {"id": image_outputs, "class": predictions_outputs}
        )
        df_predictions.drop_duplicates(subset=["id", "class"], inplace=True)

        unique_indices = df_predictions.index
        predictions_outputs = [predictions_outputs[i] for i in unique_indices]
        image_outputs = [image_outputs[i] for i in unique_indices]
        all_logits = [all_logits[i] for i in unique_indices]

        sorted_indices = [file_to_index[file_name] for file_name in image_outputs]
        sorted_image_outputs = [""] * len(test_files)
        sorted_predictions_outputs = [""] * len(test_files)
        sorted_logits = [""] * len(test_files)
        for i in range(len(sorted_indices)):
            sorted_image_outputs[sorted_indices[i]] = image_outputs[i]
            sorted_predictions_outputs[sorted_indices[i]] = predictions_outputs[i]
            sorted_logits[sorted_indices[i]] = all_logits[i]

        log.info(f"Number of test images: {len(image_outputs)}")
        log.info(f"Number of predictions: {len(predictions_outputs)}")
        log.info(f"Number of Logits: {len(all_logits)}")

        return sorted_image_outputs, sorted_predictions_outputs, sorted_logits, text_features.cpu().numpy()


