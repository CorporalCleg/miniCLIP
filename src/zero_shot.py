import os
import yaml
from pathlib import Path
import cv2
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import albumentations as A
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

from model import CLIPModel

from dataset import CLIPDataset, build_loaders
from config import cfg




def parse_yolov8_to_dataframe(dataset_path, yaml_file):
    """
    Parse YOLOv8 dataset (with nested images/labels) to DataFrame with columns: 
    ['image', 'class', 'caption']
    
    Args:
        dataset_path: Path to YOLOv8 dataset directory
        yaml_file: Path to dataset YAML file (contains class names)
        
    Returns:
        pandas DataFrame with image paths, class IDs, and class name captions
    """
    # Load class names from YAML file
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    class_names = data['names']
    
    # Prepare DataFrame columns
    records = []
    
    # Get paths to splits
    dataset_path = Path(dataset_path)
    
    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        
        # Skip if split doesn't exist
        if not split_path.exists():
            continue
            
        # Get paths to images and labels for this split
        images_dir = split_path / 'images'
        labels_dir = split_path / 'labels'
        
        # Skip if no images directory
        if not images_dir.exists():
            continue
            
        # Process each image in the split
        for image_file in images_dir.glob('*.*'):
            if image_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            # Corresponding label file
            label_file = labels_dir / f"{image_file.stem}.txt"
            
            # Skip if no label file exists
            if not label_file.exists():
                continue
                
            # Read label file
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            # Process each object in the image
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    caption = class_names[class_id]
                    
                    records.append({
                        'image': str(image_file),
                        'class': class_id,
                        'caption': f"a photo of a {caption}"  # CLIP-style caption
                    })
    
    # Create DataFrame
    df = pd.DataFrame(records)
    return df



def get_image_embeddings(valid_df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(cfg.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    model = CLIPModel().to(cfg.device)
    model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    model.eval()
    
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(cfg.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)

def classify_images(model, image_embeddings, cls_captions, temperature=1.0):
    """
    Classify image embeddings based on text captions using CLIP model.
    
    Args:
        model: CLIP model
        image_embeddings: Tensor of shape (num_images, embedding_dim)
        cls_captions: List of text captions/classes (num_classes)
        temperature: Temperature parameter for scaling logits
        
    Returns:
        Tensor of shape (num_images, num_classes) with classification logits
    """
    # Tokenize all class captions
    tokenizer = DistilBertTokenizer.from_pretrained(cfg.text_tokenizer)
    encoded_query = tokenizer(cls_captions, padding=True, truncation=True, return_tensors="pt")
    
    # Move to device
    batch = {
        key: torch.tensor(values).to(cfg.device)
        for key, values in encoded_query.items()
    }
    
    # Get text embeddings for all classes
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], 
            attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    
    # Normalize embeddings
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    
    # Compute logits (similarity scores)
    logits = (text_embeddings_n @ image_embeddings_n.T) / temperature
    
    # Return logits transposed to shape (num_images, num_classes)
    return logits.T

if __name__ == '__main__':

    dataset_path = '../zero_shot/DATASET.v2i.yolov8'
    yaml_file = '../zero_shot/DATASET.v2i.yolov8/data.yaml'
    path2ckpt = "../ckpt/best.pt"

    df = parse_yolov8_to_dataframe(dataset_path, yaml_file)
    model, image_embeddings = get_image_embeddings(df, path2ckpt)

    cls_captions = ["a photo of a bird", "a photo of a dog"]

    logits = classify_images(model, image_embeddings, cls_captions)
    probs = F.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=1).cpu().numpy()

    accuracy = np.mean(df['class'] == pred)

    print(f'Accuracy: {accuracy}')

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 9))
        

    for i, img_name in enumerate(df['image'][:9]):
        img = plt.imread(img_name)
        axs[i // 3, i % 3].imshow(img)
        axs[i // 3, i % 3].set_title(f"prediction: {cls_captions[pred[i]].split()[-1]}/ ground truth: {df['caption'][i].split()[-1]}")
        axs[i // 3, i % 3].set_axis_off()

    plt.savefig('pred_vs_gt.png')