#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:38:09 2023
@author: rehush
"""

import argparse
import os

import numpy as np
import pandas as pd
import laspy

import torch
import torch.utils.data

import MinkowskiEngine as ME
from minkowskifcnn import MinkowskiFCNN


from data import rotate_xyz, get_species_names

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/test")
parser.add_argument("--out_file", type=str, default="data/test_prediction.csv")
parser.add_argument("--net_weights", type=str, default="model/treespecies.pth")
parser.add_argument("--n_iterations", type=int, default=50)
parser.add_argument("--voxel_size", type=float, default=0.1)


def sample_xyz(xyz, n_points = 16384):
    if xyz.shape[0] >= n_points:
        np.random.shuffle(xyz)
        xyz = xyz[: n_points]
    else:
        delta_n = n_points - xyz.shape[0]
        ind = np.random.randint(xyz.shape[0], size=delta_n)
        xyz_sub = xyz[ind, : ]
        xyz = np.concatenate((xyz, xyz_sub), axis=0)
    return xyz


def most_frequent(List):
    counter = 0
    num = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i 
    return num


def predict(net, config):
    fls = os.listdir(config.data)
    treeid = []
    pred_labels = []
    spec_name = []
    
    for i in fls:
        print(i)
        # read las file, get coordinates
        las = laspy.read(config.data+'/'+i)
        xyz_ = np.vstack((las.x, las.y, las.z)).transpose()
        
        if xyz_.shape[0] > 200:
            preds = []
            for j in range(config.n_iterations):
            
                # preprocess (subsample, rotate)
                xyz = xyz_.copy()
                xyz = sample_xyz(xyz)
                xyz = rotate_xyz(xyz)
                
                # create input TensorField
                feat = xyz.copy()
                
                xyz = torch.from_numpy(xyz)
                feat = torch.from_numpy(feat)
                
                input = ME.TensorField(
                    coordinates=ME.utils.batched_coordinates([xyz / config.voxel_size], 
                                                             dtype=torch.float32),
                    features=feat.to(torch.float32),
                    device=device,
                    )
        
                # predict label
                logit = net(input)
                pred = torch.argmax(logit, 1)
                pred = pred.cpu().numpy()
            
                preds.append(pred)
        
            preds = np.concatenate(preds)
            
            # get the most frequent label 
            preds = preds.tolist()
            pred_fin = most_frequent(preds)
            pred_labels.append(pred_fin)
                
            treeid.append(int(i.split('.')[0]))
            spec_name.append(get_species_names()[pred_fin])
    
    # output file
    df = pd.DataFrame({'treeID': treeid,
                       'predicted_species': spec_name})
    df.to_csv(config.out_file, sep=',', mode='w', index =False) 
    return pred_labels, treeid
        

        
        
if __name__ == "__main__":
    config = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # import weights
    net = MinkowskiFCNN(3, 33).to(device)
    net_dict = torch.load(config.net_weights)
    net.load_state_dict(net_dict["state_dict"])
    net.eval()
    
    predict(net, config=config)


