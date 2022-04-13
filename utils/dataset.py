import os
import sys
import time
import h5py
import torch
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from prefetch_generator import background

sys.path.append(".")
from utils.config import CONF

class ScannetDatasetAllScene():
    def __init__(self, phase, scene_list, num_classes=21, npoints=32768 * 2, is_weighting=True, use_color=False, use_normal=False):
        self.phase = phase
        assert phase in ["train", "val", "test"]
        self.scene_list = scene_list
        self.num_classes = num_classes
        self.npoints = npoints
        self.is_weighting = is_weighting
        self.use_color = use_color
        self.use_normal = use_normal
        self.chunk_data = {} # init in generate_chunks()

        self._prepare_weights()

    def _prepare_weights(self):
        self.scene_data = {}
        self.multiview_data = {}
        scene_points_list = []
        semantic_labels_list = []
        
        for scene_id in tqdm(self.scene_list):
            scene_data = np.load(CONF.SCANNETV2_FILE.format(scene_id))
            label = scene_data[:, 10]

            # append
            scene_points_list.append(scene_data)
            semantic_labels_list.append(label)
            self.scene_data[scene_id] = scene_data

        if self.is_weighting:
            labelweights = np.zeros(self.num_classes)
            for seg in semantic_labels_list:
                tmp,_ = np.histogram(seg,range(self.num_classes + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        else:
            self.labelweights = np.ones(self.num_classes)

    @background()
    def __getitem__(self, index):
        start = time.time()

        # load chunks
        scene_id = self.scene_list[index]
        scene_data = self.chunk_data[scene_id]
        # unpack
        point_set = scene_data[:, :3] # include xyz by default
        rgb = scene_data[:, 3:6] / 255. # normalize the rgb values to [0, 1]
        normal = scene_data[:, 6:9]
        label = scene_data[:, 10].astype(np.int32)
        
        if self.use_color:
            point_set = np.concatenate([point_set, rgb], axis=1)

        if self.use_normal:
            point_set = np.concatenate([point_set, normal], axis=1)

        if self.phase == "train":
            point_set = self._augment(point_set)
        
        # prepare mask
        curmin = np.min(point_set, axis=0)[:3]
        curmax = np.max(point_set, axis=0)[:3]
        mask = np.sum((point_set[:, :3] >= (curmin - 0.01)) * (point_set[:, :3] <= (curmax + 0.01)), axis=1) == 3
        sample_weight = self.labelweights[label]
        sample_weight *= mask

        fetch_time = time.time() - start

        point_set = torch.FloatTensor(point_set)
        label = torch.LongTensor(label)
        sample_weight = torch.FloatTensor(sample_weight)

        return point_set, label, sample_weight, fetch_time

    def __len__(self):
        return len(self.scene_list)

    def _augment(self, point_set):
        # translate the chunk center to the origin
        center = np.mean(point_set[:, :3], axis=0)
        coords = point_set[:, :3] - center

        p = np.random.choice(np.arange(0.01, 1.01, 0.01), size=1)[0]
        if p < 1 / 8:
            # random translation
            coords = self._translate(coords)
        elif p >= 1 / 8 and p < 2 / 8:
            # random rotation
            coords = self._rotate(coords)
        elif p >= 2 / 8 and p < 3 / 8:
            # random scaling
            coords = self._scale(coords)
        elif p >= 3 / 8 and p < 4 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
        elif p >= 4 / 8 and p < 5 / 8:
            # random translation
            coords = self._translate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 5 / 8 and p < 6 / 8:
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 6 / 8 and p < 7 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        else:
            # no augmentation
            pass

        # translate the chunk center back to the original center
        coords += center
        point_set[:, :3] = coords

        return point_set

    def _translate(self, point_set):
        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]

        coords = point_set[:, :3]
        coords += [x_factor, y_factor, z_factor]
        point_set[:, :3] = coords

        return point_set

    def _rotate(self, point_set):
        coords = point_set[:, :3]

        # x rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Rx = np.array(
            [[1, 0, 0],
             [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta), np.cos(theta)]]
        )

        # y rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Ry = np.array(
            [[np.cos(theta), 0, np.sin(theta)],
             [0, 1, 0],
             [-np.sin(theta), 0, np.cos(theta)]]
        )

        # z rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Rz = np.array(
            [[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1]]
        )

        # rotate
        R = np.matmul(np.matmul(Rz, Ry), Rx)
        coords = np.matmul(R, coords.T).T

        # dump
        point_set[:, :3] = coords

        return point_set

    def _scale(self, point_set):
        # scaling factors
        factor = np.random.choice(np.arange(0.95, 1.051, 0.001), size=1)[0]

        coords = point_set[:, :3]
        coords *= [factor, factor, factor]
        point_set[:, :3] = coords

        return point_set

    def generate_chunks(self):
        """
            note: must be called before training
        """

        print("generate new chunks for {}...".format(self.phase))
        for scene_id in tqdm(self.scene_list):
            scene = self.scene_data[scene_id]
            semantic = scene[:, 10].astype(np.int32)
            
            if len(semantic) < self.npoints:
                sel_indic = np.random.choice(len(semantic), self.npoints, replace=True)
            else:
                sel_indic = np.random.choice(len(semantic), self.npoints, replace=False)

            cur_point_set = scene[sel_indic]

            self.chunk_data[scene_id] = cur_point_set
            
        print("done!\n")


def collate_random(data):
    '''
    for ScannetDataset: collate_fn=collate_random

    return: 
        coords               # torch.FloatTensor(B, N, 3)
        feats                # torch.FloatTensor(B, N, 3)
        semantic_segs        # torch.FloatTensor(B, N)
        sample_weights       # torch.FloatTensor(B, N)
        fetch_time           # float
    '''

    # load data
    (
        point_set, 
        semantic_seg, 
        sample_weight,
        fetch_time 
    ) = zip(*data)

    ## for point transformer
    offset, count = [], 0
    for item in point_set:
        count += item.shape[0]
        offset.append(count)
    offset = torch.IntTensor(offset)

    point_set = torch.cat(point_set)
    semantic_seg = torch.cat(semantic_seg)
    sample_weight = torch.cat(sample_weight)


    # split points to coords and feats
    coords = point_set[:, :3]
    feats = point_set[:, 3:]

    # pack
    batch = (
        coords,             # (B * N, 3)
        feats,              # (B * N, 3)
        semantic_seg,      # (B * N)
        sample_weight,     # (B * N)
        sum(fetch_time),   # float
        offset,            # (B)
    )

    return batch
