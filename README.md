# Install

```bash
cd lib/pointops/
python setup.py install
```

# Usage

## change the path

Before running this code, you must check the path parameters defined in `utils/config.py`.

## preprocess scannet scenes

Parse the ScanNet data into `*.npy` files and save them in `SCANNET_DIR/preprocessing/scannet_scenes/`

```bash
python preprocessing/collect_scannet_scenes.py
```

Note: you can comment line 88 ~ 90 in `preprocessing/collect_scannet_scenes.py` to process all scenes. 

Sanity check: Don't forget to visualize the preprocessed scenes to check the consistency

```bash
python preprocessing/visualize_prep_scene.py --scene_id <scene_id>
```

The visualized `<scene_id>.ply` is stored in `preprocessing/label_point_clouds/` - Drag that file into MeshLab for visualization.

## Trian with chunked scenes

### Setting

Train-test split follows the [Pointnet2.ScanNet](https://github.com/daveredrum/Pointnet2.ScanNet)

### Trian

```bash
python scripts/train_partial_scene.py --use_color --tag POINTTRANS_C_N8192 --epoch 200 --npoint 8192
```

### Visualize

```bash
python scripts/visualize_partial_scene.py --folder ${EXP_STAMP} --use_color --npoints 8192 --scene_id scene0654_00
```

The results will be saved in your `CONF.OUTPUT_ROOT` folder. Some results are visualizing as follows:

<center>
<img src="./img/scene0000_00.png" width=200px> <img src="./img/scene0652_00.png" width=200px>
</center>

## Train with complete scenes

### Setting

Train-test split follows the [Pointnet2.ScanNet](https://github.com/daveredrum/Pointnet2.ScanNet)

### Trian

```bash
python scripts/train_complete_scene.py --use_color --tag POINTTRANS_C_N32768 --epoch 200 --npoint 32768
```

### Visualize

```bash
python scripts/visualize_complete_scene.py --folder ${EXP_STAMP} --use_color --npoints 32768 --scene_id scene0654_00
```

# References

If you use this code, please cite the follow two papers:

```txt
@inproceedings{zhao2021point,
  title={Point transformer},
  author={Zhao, Hengshuang and Jiang, Li and Jia, Jiaya and Torr, Philip HS and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16259--16268},
  year={2021}
}
```

```txt
@inproceedings{dai2017scannet,
  title={Scannet: Richly-annotated 3d reconstructions of indoor scenes},
  author={Dai, Angela and Chang, Angel X and Savva, Manolis and Halber, Maciej and Funkhouser, Thomas and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5828--5839},
  year={2017}
}
```

# Acknowledgements

* [Pointnet2.ScanNet](https://github.com/daveredrum/Pointnet2.ScanNet)

* [point-transformer](https://github.com/POSTECH-CVLab/point-transformer/tree/8d2a38998f1ed8cd6d03fe1b671440724aa269c8)