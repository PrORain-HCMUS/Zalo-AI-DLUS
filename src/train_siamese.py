import torch
import numpy as np
from PIL import Image
from torch import nn
import os
import timm
import random
from ultralytics import YOLO
import yaml
from tqdm import tqdm
from src.models.siamese import Backbone, FPNWrapper, AttentionPooling, PrototypeExtractor
from src.utils import xywh_norm_to_xyxy_abs, str2list
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

class ContrastiveDetectionDataset(Dataset):
    def __init__(self, support_data, query_data, transform,
                 bg_iou_threshold=0.2, max_bg_attempts=10):

        self.support_data = support_data
        self.query_data = query_data
        self.transform = transform
        self.bg_iou_threshold = bg_iou_threshold
        self.max_bg_attempts = max_bg_attempts

        # --- Pre-process query data ---
        self.flat_query_annotations = []
        self.query_images_by_path = {}

        for query_img_info in self.query_data:
            img_path = query_img_info['img_path']
            annotations = query_img_info['annotations']

            if img_path not in self.query_images_by_path:
                self.query_images_by_path[img_path] = []

            for ann in annotations:
                self.flat_query_annotations.append({
                    'img_path': img_path,
                    'box': ann['box'],
                    'label': ann['label']
                })
                self.query_images_by_path[img_path].append(ann)

        self.all_labels = list(support_data.keys())

    def __len__(self):
        return len(self.flat_query_annotations)

    # ### FIX: Added IoU Helper function
    @staticmethod
    def calculate_iou(box1, box2):
        # Box format: [x1, y1, x2, y2]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection
        return intersection / (union + 1e-6)

    def __getitem__(self, index):

        # 1. Get Anchor
        anchor_info = self.flat_query_annotations[index]
        anchor_label = anchor_info['label']
        anchor_img_path = anchor_info['img_path']
        anchor_box = anchor_info['box']

        ref_path = random.choice(self.support_data[anchor_label])
        ref_image = Image.open(ref_path).convert('RGB')


        if random.random() < 0.5:
            label = 1.0
            search_image = Image.open(anchor_img_path).convert('RGB')
            search_patch = search_image.crop(anchor_box)
        else:
            label = 0.0
            search_image = None

            if random.random() < 0.5:
                while True:
                    neg_info = random.choice(self.flat_query_annotations)
                    if neg_info['label'] != anchor_label:
                        break

                search_image = Image.open(neg_info['img_path']).convert('RGB')
                search_patch = search_image.crop(neg_info['box'])

            else:
                search_image = Image.open(anchor_img_path).convert('RGB')
                img_w, img_h = search_image.size

                # ### FIX: Ensure these are integers for random.randint
                all_gt_boxes = [ann['box'] for ann in self.query_images_by_path[anchor_img_path]]

                anchor_w = int(anchor_box[2] - anchor_box[0])
                anchor_h = int(anchor_box[3] - anchor_box[1])

                found_negative = False
                for _ in range(self.max_bg_attempts):
                    # ### FIX: Cast max range to int to prevent TypeError
                    max_x = int(max(0, img_w - anchor_w - 1))
                    max_y = int(max(0, img_h - anchor_h - 1))

                    # If image is smaller than crop (edge case), break
                    if max_x == 0 or max_y == 0:
                        break

                    neg_x1 = random.randint(0, max_x)
                    neg_y1 = random.randint(0, max_y)

                    neg_x2 = neg_x1 + anchor_w
                    neg_y2 = neg_y1 + anchor_h
                    neg_box = [neg_x1, neg_y1, neg_x2, neg_y2]

                    # Check IoU
                    max_iou = 0.0
                    for gt_box in all_gt_boxes:
                        iou = self.calculate_iou(neg_box, gt_box)
                        max_iou = max(max_iou, iou)

                    if max_iou <= self.bg_iou_threshold:
                        search_patch = search_image.crop(neg_box)
                        found_negative = True
                        break

                # Fallback if background mining failed
                if not found_negative:
                    while True:
                        neg_info = random.choice(self.flat_query_annotations)
                        if neg_info['label'] != anchor_label:
                            break
                    search_image = Image.open(neg_info['img_path']).convert('RGB')
                    search_patch = search_image.crop(neg_info['box'])

        # 4. Transforms
        if self.transform:
            ref_image = self.transform(ref_image)
            search_patch = self.transform(search_patch)

        return ref_image, search_patch, torch.tensor(label, dtype=torch.float32)


def prepare_data(args):
    
    gd_img_dict = {}
    idx = 0
    support_folder = "data/observing/train/samples/"
    support_folder = args.support_data
    for fol in os.listdir(support_folder):
        gd_img_dict[idx] = []
        for img_name in os.listdir(support_folder + fol + "/object_images/"):
            img_path =support_folder + fol+ "/object_images/" + img_name
            gd_img_dict[idx].append(img_path)
        idx += 1
    img_folder = "/kaggle/input/zaloai-aero/data_one_class/data_one_class/train/images/"
    ground_truth_folder = "/kaggle/input/zaloai-aero/labels_n_class/labels/"
    img_folder = args.image_folder
    ground_truth_folder = args.labels
    ground_truth_labels = os.listdir(ground_truth_folder)
    query_data = []
    for gd_label in ground_truth_labels:
        # image_id = gd_label.split('.')[0]
        with open(ground_truth_folder + "/" + gd_label, "r") as f:
                gd_bboxs = [str2list(a) for a in f.readlines()]
        img_path = img_folder + "/" + gd_label.split(".")[0] + ".jpg"
        gd_bboxs_dict = {
            "img_path": img_path,
            "annotations": [],
        }
        for gd_bbox in gd_bboxs:
            gd_bboxs_dict["annotations"].append({
                "box": xywh_norm_to_xyxy_abs(gd_bbox[1:]),
                "label": int(gd_bbox[0])
            })
        query_data.append(gd_bboxs_dict)
    return query_data, gd_img_dict


def prepare_model(query_data, gd_img_dict, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        "hf_hub:timm/mobilenetv4_conv_medium.e500_r224_in1k",              
        pretrained=True,
        features_only=True,
        out_indices=(1, 2, 4)  # Number of output classes
    ).to(device)
    model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    len_train = int(0.9 * len(query_data))

    t_set, v_set = random_split(query_data, [len_train, len(query_data) - len_train])
    train_set = ContrastiveDetectionDataset(gd_img_dict, t_set, transform=transforms)
    val_set = ContrastiveDetectionDataset(gd_img_dict, v_set, transform=transforms)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True)
    backbone = Backbone(model)
    fpn = FPNWrapper(backbone)
    att = AttentionPooling()
    prototypeExtractor = PrototypeExtractor(fpn, att)
    return train_loader, val_loader, prototypeExtractor

def train_model(prototypeExtractor, train_loader, val_loader, args):
    num_epochs = args.epochs
    lr = 3e-4
    batch_size = 1 
    best_val_loss = float('inf') #
    min_delta = 1e-4
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prototypeExtractor.to(device)
    loss_fn = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(prototypeExtractor.parameters(), lr=lr)

    start_epoch = 0


    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs, 
        last_epoch=start_epoch - 1
    )

    patience = 12
    current_patient = 0

    os.makedirs("runs/siamese", exist_ok=True)

    # 2. Training Loop
    global_step = 0

    for epoch in range(start_epoch, num_epochs):

        # ==========================
        #       TRAINING PHASE
        # ==========================
        prototypeExtractor.train()

        train_loss = 0.0
        train_pos_sim = 0.0
        train_neg_sim = 0.0
        train_pos_count = 0
        train_neg_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch in pbar:
            input1 = batch[0].to(device)
            input2 = batch[1].to(device)
            label = batch[2].float().to(device)
            # Target for Loss (-1/1)
            target = torch.where(label == 0, torch.tensor(-1.0).to(device), label)

            # Forward pass
            proto_a = prototypeExtractor(input1)
            proto_b = prototypeExtractor(input2)

            loss = loss_fn(proto_a, proto_b, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # --- FIX: Vectorized Metrics (No more loops) ---
            with torch.no_grad():
                # 1. Calculate Cosine Similarity
                # .view(-1) ensures we have a flat vector, even if batch_size=1
                sims = torch.cosine_similarity(proto_a, proto_b, dim=1).view(-1)
                label_flat = label.view(-1)

                # 2. Create Boolean Masks
                pos_mask = (label_flat == 1.0)
                neg_mask = (label_flat == 0.0) # or ~pos_mask if labels are strictly 0/1

                # 3. Aggregate (Sum) based on masks
                # We use checks to avoid summing empty tensors if a batch has no pos or no neg samples
                if pos_mask.any():
                    train_pos_sim += sims[pos_mask].sum().item()
                    train_pos_count += pos_mask.sum().item()

                if neg_mask.any():
                    train_neg_sim += sims[neg_mask].sum().item()
                    train_neg_count += neg_mask.sum().item()

            # Log Step Metrics

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1
        scheduler.step()
        # ==========================
        #      VALIDATION PHASE
        # ==========================
        prototypeExtractor.eval()

        val_loss = 0.0
        val_pos_sim = 0.0
        val_neg_sim = 0.0
        val_pos_count = 0
        val_neg_count = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

            for batch in val_pbar:
                input1 = batch[0].to(device)
                input2 = batch[1].to(device)
                label = batch[2].float().to(device)
                target = torch.where(label == 0, torch.tensor(-1.0).to(device), label)

                proto_a = prototypeExtractor(input1)
                proto_b = prototypeExtractor(input2)

                loss = loss_fn(proto_a, proto_b, target)
                val_loss += loss.item()

                # --- FIX: Same Vectorized Logic for Validation ---
                sims = torch.cosine_similarity(proto_a, proto_b, dim=1).view(-1)
                label_flat = label.view(-1)

                pos_mask = (label_flat == 1.0)
                neg_mask = (label_flat == 0.0)

                if pos_mask.any():
                    val_pos_sim += sims[pos_mask].sum().item()
                    val_pos_count += pos_mask.sum().item()

                if neg_mask.any():
                    val_neg_sim += sims[neg_mask].sum().item()
                    val_neg_count += neg_mask.sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_pos = train_pos_sim / (train_pos_count + 1e-6)
        avg_train_neg = train_neg_sim / (train_neg_count + 1e-6)

        # Calculate Val Averages
        avg_val_loss = val_loss / len(val_loader)
        avg_val_pos = val_pos_sim / (val_pos_count + 1e-6)
        avg_val_neg = val_neg_sim / (val_neg_count + 1e-6)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Pos Sim: {avg_val_pos:.4f} | Val Neg Sim: {avg_val_neg:.4f}")

        if avg_val_loss > best_val_loss:
            current_patient += 1
            if current_patient >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break
        else:
            current_patient = 0

        # Log Epoch Metrics to WandB
        # ==========================
        #      CHECKPOINTING
        # ==========================

        # 1. Save latest model
        ckpt = {
            'epoch': epoch,
            'model_state_dict': prototypeExtractor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        torch.save(ckpt, "runs/siamese/last.pth")

        # 2. Save best model (if validation loss improved)
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            current_patient = 0
            torch.save(ckpt, "runs/siamese/best.pth")
            print(f"--> New Best Model Saved! (Val Loss: {best_val_loss:.4f})")
        else:
            current_patient += 1
            if current_patient >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Train YOLO11 for AeroEyes')
    parser.add_argument('--support_data', type=str, required=True, help='Path to support data folder')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to training images folder')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels folder')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    
    args = parser.parse_args()

    query_data, gd_img_dict = prepare_data(args)
    train_loader, val_loader, prototypeExtractor = prepare_model(query_data, gd_img_dict, args)
    train_model(prototypeExtractor, train_loader, val_loader, args)
