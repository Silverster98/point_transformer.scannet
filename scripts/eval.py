import numpy as np
import torch

from utils.config import CONF
from utils.pc_util import point_cloud_label_to_surface_voxel_label_fast

def get_scene_list(path):
    scene_list = []
    with open(path) as f:
        for scene_id in f.readlines():
            scene_list.append(scene_id.strip())

    scene_list = sorted(scene_list, key=lambda x: int(x.split("_")[0][5:]))

    return scene_list

def forward(args, model, coords, feats):
    pred = []
    coord_chunk, feat_chunk = torch.split(coords.squeeze(0), args.batch_size, 0), torch.split(feats.squeeze(0), args.batch_size, 0)
    # print('tag', len(coord_chunk), coord_chunk[0].shape) <1, chunk_size, N, C>
    # print('tag', len(feat_chunk), feat_chunk[0].shape)
    assert len(coord_chunk) == len(feat_chunk)
    for coord, feat in zip(coord_chunk, feat_chunk):
        output = model(torch.cat([coord, feat], dim=2))
        pred.append(output)

    pred = torch.cat(pred, dim=0).unsqueeze(0) # (1, CK, N, C)
    # print(pred.shape) <1, chunk_size, N>
    outputs = pred.max(3)[1]

    return outputs

def filter_points(coords, preds, targets, weights):
    assert coords.shape[0] == preds.shape[0] == targets.shape[0] == weights.shape[0]
    coord_hash = [hash(str(coords[point_idx][0]) + str(coords[point_idx][1]) + str(coords[point_idx][2])) for point_idx in range(coords.shape[0])]
    _, coord_ids = np.unique(np.array(coord_hash), return_index=True)
    coord_filtered, pred_filtered, target_filtered, weight_filtered = coords[coord_ids], preds[coord_ids], targets[coord_ids], weights[coord_ids]

    return coord_filtered, pred_filtered, target_filtered, weight_filtered

def compute_acc(coords, preds, targets, weights):
    coords, preds, targets, weights = filter_points(coords, preds, targets, weights)
    seen_classes = np.unique(targets)
    mask = np.zeros(CONF.NUM_CLASSES)
    mask[seen_classes] = 1

    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(CONF.NUM_CLASSES)]
    total_correct_class = [0 for _ in range(CONF.NUM_CLASSES)]

    total_correct_vox = 0
    total_seen_vox = 0
    total_seen_class_vox = [0 for _ in range(CONF.NUM_CLASSES)]
    total_correct_class_vox = [0 for _ in range(CONF.NUM_CLASSES)]

    labelweights = np.zeros(CONF.NUM_CLASSES)
    labelweights_vox = np.zeros(CONF.NUM_CLASSES)

    correct = np.sum(preds == targets) # evaluate only on 20 categories but not unknown
    total_correct += correct
    total_seen += targets.shape[0]
    tmp,_ = np.histogram(targets,range(CONF.NUM_CLASSES+1))
    labelweights += tmp
    for l in seen_classes:
        total_seen_class[l] += np.sum(targets==l)
        total_correct_class[l] += np.sum((preds==l) & (targets==l))

    _, uvlabel, _ = point_cloud_label_to_surface_voxel_label_fast(coords, np.concatenate((np.expand_dims(targets,1),np.expand_dims(preds,1)),axis=1), res=0.02)
    total_correct_vox += np.sum(uvlabel[:,0]==uvlabel[:,1])
    total_seen_vox += uvlabel[:,0].shape[0]
    tmp,_ = np.histogram(uvlabel[:,0],range(CONF.NUM_CLASSES+1))
    labelweights_vox += tmp
    for l in seen_classes:
        total_seen_class_vox[l] += np.sum(uvlabel[:,0]==l)
        total_correct_class_vox[l] += np.sum((uvlabel[:,0]==l) & (uvlabel[:,1]==l))

    pointacc = total_correct / float(total_seen)
    voxacc = total_correct_vox / float(total_seen_vox)

    labelweights = labelweights.astype(np.float32)/np.sum(labelweights.astype(np.float32))
    labelweights_vox = labelweights_vox.astype(np.float32)/np.sum(labelweights_vox.astype(np.float32))
    caliweights = labelweights_vox
    voxcaliacc = np.average(np.array(total_correct_class_vox)/(np.array(total_seen_class_vox,dtype=np.float)+1e-8),weights=caliweights)

    pointacc_per_class = np.zeros(CONF.NUM_CLASSES)
    voxacc_per_class = np.zeros(CONF.NUM_CLASSES)
    for l in seen_classes:
        pointacc_per_class[l] = total_correct_class[l]/(total_seen_class[l] + 1e-8)
        voxacc_per_class[l] = total_correct_class_vox[l]/(total_seen_class_vox[l] + 1e-8)

    return pointacc, pointacc_per_class, voxacc, voxacc_per_class, voxcaliacc, mask

def compute_miou(coords, preds, targets, weights):
    coords, preds, targets, weights = filter_points(coords, preds, targets, weights)
    seen_classes = np.unique(targets)
    mask = np.zeros(CONF.NUM_CLASSES)
    mask[seen_classes] = 1

    pointmiou = np.zeros(CONF.NUM_CLASSES)
    voxmiou = np.zeros(CONF.NUM_CLASSES)

    uvidx, uvlabel, _ = point_cloud_label_to_surface_voxel_label_fast(coords, np.concatenate((np.expand_dims(targets,1),np.expand_dims(preds,1)),axis=1), res=0.02)
    for l in seen_classes:
        target_label = np.arange(targets.shape[0])[targets==l]
        pred_label = np.arange(preds.shape[0])[preds==l]
        num_intersection_label = np.intersect1d(pred_label, target_label).shape[0]
        num_union_label = np.union1d(pred_label, target_label).shape[0]
        pointmiou[l] = num_intersection_label / (num_union_label + 1e-8)

        target_label_vox = uvidx[(uvlabel[:, 0] == l)]
        pred_label_vox = uvidx[(uvlabel[:, 1] == l)]
        num_intersection_label_vox = np.intersect1d(pred_label_vox, target_label_vox).shape[0]
        num_union_label_vox = np.union1d(pred_label_vox, target_label_vox).shape[0]
        voxmiou[l] = num_intersection_label_vox / (num_union_label_vox + 1e-8)

    return pointmiou, voxmiou, mask
