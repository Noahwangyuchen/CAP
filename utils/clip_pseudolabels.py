import logging
import os
import pickle

import clip
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from data.text_augmentations import get_text_aug

log = logging.getLogger(__name__)


def compute_pseudo_labels(
    k,
    template,
    filepaths,
    classnames,
    transform,
    clip_model,
    label_to_idx,
    device,
    filename,
):
    prompts = [f"{template}{' '.join(i.split('_'))}" for i in classnames]
    text = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    all_file_paths = []
    all_image_features = []
    all_probs = []

    log.info(f"Compute pseudo-labeles")
    for i, image_path in enumerate(tqdm(filepaths)):
        img = Image.open(image_path).convert("RGB")
        img = transform(img).to(device).unsqueeze(0)
        with torch.no_grad():
            image_feature = clip_model.encode_image(img)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            logit_scale = clip_model.logit_scale.exp()
            logits_per_image = logit_scale * image_feature @ text_features.t()
            probs = logits_per_image.softmax(dim=-1)
            all_file_paths.append(image_path)
            all_image_features.append(image_feature[0].cpu().numpy())
            all_probs.append(probs[0].cpu().numpy())

    log.info(f"Save pseudo-labels to {filename}, {len(all_file_paths)} images")
    with open(filename, "wb") as f:
        pickle.dump({"filepaths": all_file_paths, "probs": all_probs, "image_features": all_image_features, "text_features": text_features.cpu().numpy()}, f)

def pseudolabel_top_k(
    config,
    data_name,
    k,
    template,
    labeled_filepaths,
    labels,
    unlabeled_filepaths,
    classnames,
    transform,
    clip_model,
    label_to_idx,
    device,
    vis_encoder,
    split_seed, 
):
    filename = f"pseudolabels/{data_name}_{vis_encoder.replace('/', '')}_pseudolabels_split_{split_seed}.pickle"
    if not os.path.exists(filename):
        compute_pseudo_labels(
            k,
            template,
            list(labeled_filepaths) + list(unlabeled_filepaths),    # compute pseudo-labels for all images
            classnames,
            transform,
            clip_model,
            label_to_idx,
            device,
            filename,
        )

    with open(filename, "rb") as f:
        content = pickle.load(f)
        all_filepaths = content["filepaths"]
        all_probs = content["probs"]
        text_features = torch.tensor(content["text_features"]).to(device)
        image_features = torch.tensor(content["image_features"]).to(device)

    top_k_leaderboard = {label_to_idx[classnames[i]]: [] for i in range(len(classnames))}
        
    for img_path, prob in zip(all_filepaths, all_probs):
        """if predicted class has empty leaderboard, or if the confidence is high
        enough for predicted class leaderboard, add the new example
        """
        # if the image is labeled, skip it in pseudo-labeling
        if img_path in labeled_filepaths or img_path not in unlabeled_filepaths:
            continue

        pred = prob.argmax()
        prob_score = prob[pred]
        if pred in top_k_leaderboard.keys() and len(top_k_leaderboard[pred]) < k:
            top_k_leaderboard[pred] = sorted(
                top_k_leaderboard[pred] + [(prob[pred], img_path)],
                reverse=True,
            )
        elif (
            pred in top_k_leaderboard.keys() and top_k_leaderboard[pred][-1][0] < prob_score
        ):  # if the confidence in predicted class "qualifies" for top-k
            # default sorting of tuples is by first element
            top_k_leaderboard[pred] = sorted(
                top_k_leaderboard[pred] + [(prob[pred], img_path)],
                reverse=True,
            )[:k]
        else:
            # sort the other classes by confidence score
            order_of_classes = sorted(
                [(prob[j], j) for j in range(len(label_to_idx)) if j != pred],
                reverse=True,
            )
            for score, index in order_of_classes:
                if index not in top_k_leaderboard.keys():
                    continue
                index_dict = index
                if len(top_k_leaderboard[index_dict]) < k:
                    top_k_leaderboard[index_dict].append((prob[index], img_path))
                elif top_k_leaderboard[index_dict][-1][0] < prob[index]:
                    # default sorting of tuples is by first element
                    top_k_leaderboard[index_dict] = sorted(
                        top_k_leaderboard[index_dict]
                        + [((prob[index], img_path))],
                        reverse=True,
                    )[:k]

    text_augs = get_text_aug(config.DATASET_NAME)
    for class_name, text_aug in text_augs.items():
        if class_name not in classnames:
            continue
        text_aug = clip_model.encode_text(clip.tokenize(text_aug).to(device)).detach()
        text_aug /= text_aug.norm(dim=-1, keepdim=True)
        similarity = clip_model.logit_scale.exp().detach() * text_aug @ image_features.t()
        top_k_indices = similarity.topk(k, dim=-1).indices.tolist()
        top_k_leaderboard[label_to_idx[class_name]] = [(similarity[0][i], all_filepaths[i]) for i in top_k_indices[0]]
            
    return top_k_leaderboard
