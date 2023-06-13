#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Apr 2023
@author: rehush
"""

import argparse
import sklearn.metrics as metrics
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

import MinkowskiEngine as ME
from data import TreeSpeciesCRS, get_class_weights, get_species_names

from common import seed_all
from minkowskifcnn import MinkowskiFCNN


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="data")
parser.add_argument("--n_points", type=int, default=16384)
parser.add_argument("--voxel_size", type=float, default=0.1)
parser.add_argument("--max_steps", type=int, default=100000)
parser.add_argument("--val_freq", type=int, default=1000)
parser.add_argument("--stat_freq", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-1)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--weights", type=str, default="treespecies.pth")
parser.add_argument("--out_dir", type=str, default="model")
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--network", type=str, choices=["minkfcnn"], default="minkfcnn")


STR2NETWORK = dict(
    minkfcnn=MinkowskiFCNN,
)


def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["label"] for d in list_data],
        dtype=torch.float32,
    )
    return {
        "coordinates": coordinates_batch,
        "features": features_batch,
        "labels": labels_batch,
    }


def create_input_batch(batch, is_minknet, device="cuda", 
                       quantization_size=0.1):
    batch["coordinates"][:, 1:] = batch["coordinates"][:, 1:] / quantization_size
    return ME.TensorField(
        coordinates=batch["coordinates"],
        features=batch["features"],
        device=device,
        )

def make_data_loader(dataset, is_minknet, config, shuffle = True):
    return DataLoader(
        dataset,
        num_workers=config.num_workers,
        shuffle=shuffle,
        collate_fn=minkowski_collate_fn,
        batch_size=config.batch_size,
    )

def plot_conf_matrix(labels, preds, class_names, cm_file): 
        cm = confusion_matrix(labels, preds, normalize = "true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(20,20))
        ax.tick_params(axis='both', which='major', labelsize=20)
        disp.plot(ax = ax, xticks_rotation='vertical', colorbar=False)
        plt.xlabel('Predicted label', fontsize=30)
        plt.ylabel('True label', fontsize=30)
        plt.tight_layout()
        plt.show()
        plt.savefig(cm_file, dpi = 300, pad_inches=5)
        

def criterion(pred, labels, smoothing=True, class_weights=None):
    """Calculate cross entropy loss, apply label smoothing if needed."""

    labels = labels.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, labels.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    elif class_weights is not None:
        loss = F.cross_entropy(pred, labels, weight = class_weights, 
                               reduction="mean", label_smoothing=0.2)
    else:
        loss = F.cross_entropy(pred, labels, reduction="mean")
    return loss


def evaluate(net, dataset, device, config, phase="val",    
             out_file = "classification_report.txt", 
             cm_file = "confusion_matrix.png"):
    is_minknet = isinstance(net, ME.MinkowskiNetwork)
    data_loader = make_data_loader(dataset, is_minknet, config, shuffle=True)

    net.eval()
    labels, preds = [], []
    with torch.no_grad():
        for batch in data_loader:
            input = create_input_batch(
                batch,
                is_minknet,
                device=device,
                quantization_size=config.voxel_size,
            )
            logit = net(input)
            
            # get predicted classes
            pred = torch.argmax(logit, 1)
            labels.append(batch["labels"].cpu().numpy())
            preds.append(pred.cpu().numpy())
            torch.cuda.empty_cache()
    
    labels = np.concatenate(labels)
    preds = np.concatenate(preds)
    
    # calculate statistics
    accuracy = metrics.accuracy_score(labels, preds)
    ballanced_accuracy_score = metrics.balanced_accuracy_score(labels, preds)
    f1_score = metrics.f1_score(labels, preds, average='macro')
    
    # get classification report 
    f_st = open(out_file, "w")
    print(metrics.classification_report(labels, preds), file=f_st)
    f_st.close()
        
    # plot confusion matrix
    plot_conf_matrix(labels = labels, preds=preds, 
                     class_names=get_species_names(), 
                     cm_file = cm_file)
    
    return accuracy, ballanced_accuracy_score, f1_score


def train(net, device, config):
    is_minknet = isinstance(net, ME.MinkowskiNetwork)
    print(is_minknet)
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=0.9,
        weight_decay=config.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_steps,
    )
    
    print(optimizer)
    print(scheduler)
    
    dataset_train = TreeSpeciesCRS(
        phase='train',
        data_root = config.data_root,
        n_points = config.n_points, 
        augment=True,
    )
    train_iter = iter(make_data_loader(dataset_train, is_minknet, config, shuffle=True))
    
    # validation data, not augmented
    dataset_val = TreeSpeciesCRS(
        phase='val',
        data_root = config.data_root,
        n_points = config.n_points, 
        augment=False
    )
    
    best_f1 = 0
    net.train()
    for i in range(config.max_steps):
        optimizer.zero_grad()
        try:
            data_dict = train_iter.next()
        except StopIteration:
            train_iter = iter(make_data_loader(dataset_train, is_minknet, config, shuffle=True))
            data_dict = train_iter.next()
        
        input = create_input_batch(
            data_dict, is_minknet, device=device, 
            quantization_size=config.voxel_size)
        logit = net(input)  
        loss = criterion(logit, data_dict["labels"].to(device), 
                         class_weights=get_class_weights().to(device))
            	
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        torch.cuda.empty_cache()

        if i % config.stat_freq == 0:
            print(f"Iter: {i}, Loss: {loss.item():.3e}")

        if i % config.val_freq == 0 and i > 0:
            accuracy, f1 = evaluate(net, dataset_val, device, config, phase="val", 
                                    out_file = config.out_dir+"/statistics_"+str(i)+".txt", 
                                    cm_file = config.out_dir+"/conf_matrix_"+str(i)+".png")
            print(f"Validation F1: {f1}. Best F1: {best_f1}")
            print(f"Validation accuracy: {accuracy}. ")
            
            if best_f1 < f1:
                best_f1 = f1
                
                # save stats and model
                f_metric = open(config.out_dir+"/best_F1.txt", "w")
                print([i, best_f1], file=f_metric)
                f_metric.close()
                torch.save(
                    {
                        "state_dict": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "curr_iter": i,
                    },
                    config.out_dir+'/'+config.weights,
                )
                
            net.train()


if __name__ == "__main__":
    config = parser.parse_args()
    seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===================Tree Species Dataset===================")
    print(f"Network: {config.network}")
    print(f"Voxel size: {config.voxel_size}")
    print(f"Number of points: {config.n_points}")
    print(f"Batch size: {config.batch_size}")
    print(f"Max. steps: {config.max_steps}")
    print(f"Validation freq: {config.val_freq}")
    
    print("=============================================\n\n")
    net = STR2NETWORK[config.network](
        in_channel=3, out_channel=33).to(device)

    train(net, device, config)

