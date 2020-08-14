# Localization network

Trains a localization network to predict probability that two frames belong to the same zone.
```
python -m build_graph.localization_network.train --dset epic --max_iter 20000 --cv_dir build_graph/localization_network/cv/epic/ --parallel 
```
Output: `build_graph/localization_network/cv/epic/ckpt_E_250_I_19250_L_0.368_A_0.834.pth`

### Requirements:
Several intermediate files are required to sample pairs to train the localization network

------
Generate features for each video clip (average ResNet152 feature for all frames). Then calculate pairwise L2 distance between features of every clip in the same kitchen.
```
python -m retrieval_network.data.generate_r152_neighbors --dset epic
```
Output dir: `build_graph/data/epic/`
Files: `avg_feats_r152.pth`, `train_pdist_r152_samek.pth`, `train_pdist_r152_samek.pth`

-------

Generate superpoint descriptors for every clip
```
bash build_graph/tools/generate_sp_descriptors.sh epic
```
Output dir: `build_graph/data/epic/descriptors/`
Files: `{clip_id}.pth` (N=28472)

-------
Use computed descriptors to generate keypoint matches between every clip pair.
```
bash build_graph/tools/generate_sp_matches.sh epic train
bash build_graph/tools/generate_sp_matches.sh epic val
```
Output dir: `build_graph/data/epic/matches/train`, `build_graph/data/epic/matches/val`
Files: `{kitchen_id}.pth` (N=31)

-------
Calculate pairwise distance between every clip in the same kitchen. The score is based on the previously estimated homographies.
```
python -m build_graph.tools.generate_homography_neighbors --dset epic
```
Output dir: `build_graph/data/epic/`
Files: `train_homography_inliers.pth`, `val_homography_inliers.pth`