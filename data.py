#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Apr 2023
@author: rehush
"""


import os
import random
import numpy as np
import pandas as pd
import math
import laspy

import torch
from torch.utils.data import Dataset


def get_species_names(file_path = 'data/tree_metadata_training_publish.csv'):
    info = pd.read_csv(file_path)
    species = info['species'].unique().tolist()
    species = np.array(species)
    return species
        

def get_class_weights():
    csv_path = 'data/tree_metadata_training_publish.csv'
    info = pd.read_csv(csv_path)
    species = info['species'].unique().tolist()
        
    # get class weights
    species_count = info.value_counts('species')
    class_weights = []
    for i in species: 
        w = round(species_count[0]/species_count[i])
        if w > 1:
            w = w-1
            class_weights.append(w) 
    return torch.tensor(class_weights + [0.01])


def rotate_xyz(xyz):
    '''
    Rotate point cloud around Z axis (using medoid of X and Y coors) by angle=angle. 
    Rotation angle will be generated randomly. 
    '''
    angle = random.randint(0,360)
    angle = math.radians(angle)
    
    ox=np.mean(xyz[:,0])
    oy=np.mean(xyz[:,1])
    
    x_ = xyz[:,0].copy()
    y_ = xyz[:,1].copy()
    x_ = x_ - ox
    y_ = y_ - oy
    
    px = x_ * math.cos(angle) - y_ * math.sin(angle)
    py = x_ * math.sin(angle) + y_ * math.cos(angle)
    
    xyz[:,0] = px + ox
    xyz[:,1] = py + oy 
    
    return xyz
            
class TreeSpeciesCRS(Dataset):
    def __init__(
        self,
        phase: str,
        data_root: str = "data",
        n_points=16384,
        augment = True,
    ):
        Dataset.__init__(self)

        self.data, self.label = self.load_data(data_root, phase, n_points)
        self.augment = augment
        self.phase = phase
        self.n_points = n_points

    def load_data(self, data_root, phase, n_points):
        csv_path = data_root+'/tree_metadata_training_publish.csv'
        info = pd.read_csv(csv_path)
        species = info['species'].unique().tolist()
        
        # get class weights
        species_count = info.value_counts('species')
        class_weights = []
        for i in species: 
            w = round(species_count[0]/species_count[i])
            if w > 1:
                w = w-1
            class_weights.append(w)

        data, labels = [], []
        assert os.path.exists(data_root), f"{data_root} does not exist"
        
        files = pd.read_csv(data_root+'/'+phase+'.txt', header=None)
        files = files.values.tolist()
        assert len(files) > 0, "No files found"
        
        for i in range(0, len(files)):
            f = files[i][0]

            # get data
            xyz_file = data_root+'/train/'+f
            if os.path.exists(xyz_file):
                # get data
                las = laspy.read(xyz_file)
                xyz = np.vstack((las.x, las.y, las.z)).transpose()
                if xyz.shape[0] > 200:
                    if xyz.shape[0] >= n_points:
                        np.random.shuffle(xyz)
                        xyz = xyz[: n_points]
                    elif xyz.shape[0] < n_points: 
                        delta_n = n_points - xyz.shape[0]
                        ind = np.random.randint(xyz.shape[0], size=delta_n)
                        xyz_sub = xyz[ind, : ]
                        xyz = np.concatenate((xyz, xyz_sub), axis=0)
               
                    data.append(xyz.astype("float32"))
                
                    # get class label
                    f_info = info[info['filename'] == '/train/'+f]
                    l = [species.index(f_info.values.tolist()[0][1])]
                    l = np.asarray(l)
                    labels.append(l.astype("int64"))
        
        data = np.stack(data, axis=0)
        labels = np.stack(labels, axis=0)
        print("Data shape")
        print(phase)
        print(data.shape)
        return data, labels

    def __getitem__(self, i: int) -> dict:
        xyz = self.data[i]
        if self.phase == "train":
            np.random.shuffle(xyz)
        if len(xyz) > self.n_points:
            xyz = xyz[: self.n_points]
        if self.augment == True:    
            xyz = rotate_xyz(point_cloud=xyz)

        feat = xyz.copy()        
        label = self.label[i]
        label = torch.from_numpy(label)
        xyz = torch.from_numpy(xyz)
        feat = torch.from_numpy(feat)
        return {
            "coordinates": xyz.to(torch.float32),
            "features": feat.to(torch.float32),
            "label": label,
        }

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return f"TreeSpeciesData(phase={self.phase}, length={len(self)}, transform={self.transform})"

