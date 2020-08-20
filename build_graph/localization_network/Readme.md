# Localization network

Trains a localization network to predict probability that two frames belong to the same zone.
```
python -m build_graph.localization_network.train --dset epic --max_iter 20000 --cv_dir build_graph/localization_network/cv/epic/ --parallel 
```
Outputs:
```
build_graph/localization_network/cv/epic/ckpt_E_250_I_19250_L_0.368_A_0.834.pth
```

### Requirements:
Several intermediate files are required to sample pairs to train the localization network. We recommend using our precomputed intermediate files as these take a large amount of time to generate.

------
**Generate features for each video clip** (average ResNet152 feature for all frames). Then calculate pairwise L2 distance between features of every clip in the same kitchen.
```
python -m retrieval_network.data.generate_r152_neighbors --dset epic
```
Outputs:
```
build_graph/data/epic/avg_feats_r152.pth
build_graph/data/epic/train_pdist_r152_samek.pth
build_graph/data/epic/val_pdist_r152_samek.pth
```

-------
**Generate superpoint descriptors for every clip**
```
bash build_graph/tools/generate_sp_descriptors.sh epic
```
Outputs:
```
build_graph/data/epic/descriptors/{clip_1}.pth --> {clip_N}.pth
``` 

-------
**Use computed descriptors to generate keypoint matches between every clip pair**
```
bash build_graph/tools/generate_sp_matches.sh epic train
bash build_graph/tools/generate_sp_matches.sh epic val
```
Outputs:
```
build_graph/data/epic/matches/train/{kitchen_1}.pth --> {kitchen_N}.pth
build_graph/data/epic/matches/val/{kitchen_1}.pth --> {kitchen_N}.pth
```

Visualize matches:
```
python -m build_graph.tools.viz_sp_matches --dset epic
```

-------
**Calculate pairwise distance between every clip in the same kitchen.** The score is based on the previously estimated homographies.
```
python -m build_graph.tools.generate_homography_neighbors --dset epic
```
Outputs:
```
build_graph/data/epic/train_homography_inliers.pth
build_graph/data/epic/val_homography_inliers.pth
```





