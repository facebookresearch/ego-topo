# Graph building

Build the graph for each video. Use `--viz` to visualize, omit to save files.
```
# Using just the frames from the video (very slow)
python -m build_graph.build --dset epic --v_id P26_01 --locnet_wts build_graph/localization_network/cv/epic/ckpt_E_250_I_19250_L_0.368_A_0.834.pth --online --frame_inp --viz

# Using precomputed localization net trunk features (slow). See requirements below
python -m build_graph.build --dset epic --v_id P26_01 --locnet_wts build_graph/localization_network/cv/epic/ckpt_E_250_I_19250_L_0.368_A_0.834.pth --online --viz

# Using precomputed localization net scores (fast). See requirements below
python -m build_graph.build --dset epic --v_id P26_01 --viz
```
Output dir: `build_graph/data/epic/graphs/`
Files: `P26_01_graph.pth` (N = #videos)

### Requirements:

Download the EPIC/EGTEA+ datasets and symlink the frames and annotations directories to the project data folder.
```
ln -s /vision/vision_users/tushar/datasets/EPIC_KITCHENS_2018/annotations/ data/epic
ln -s /vision/vision_users/tushar/datasets/EPIC_KITCHENS_2018/frames_rgb_flow/ data/epic
ln -s /vision/vision_users/tushar/datasets/EGTEA+/annotation/ data/gtea/
ln -s /vision/vision_users/tushar/datasets/EGTEA+/frames/ data/gtea/
```
-------

Train a localization network (see `build_graph/localization_network/Readme.md`) to generate a model checkpoint:
`build_graph/localization_network/cv/epic/ckpt_E_250_I_19250_L_0.368_A_0.834.pth`
-------

(Optional) Precompute localization network features for all frames in the dataset to speed up graph construction. Recommended.
```
 python -m build_graph.tools.generate_locnet_features --load build_graph/localization_network/cv/epic/ckpt_E_250_I_19250_L_0.368_A_0.834.pth --dset epic
```
Output dir: `build_graph/data/epic/`
Files: `locnet_trunk_feats.pth`

-------
(Optional) Precompute pairwise localization net similarity matrix per video to speed up graph construction. Recommended.
```
python -m build_graph.tools.generate_pairwise_locnet_scores --load build_graph/localization_network/cv/epic/head.pth --dset epic --parallel --v_id P26_01
```
Output dir: `build_graph/data/epic/locnet_pairwise_scores`
Files: `P26_01_scores.pth` (N = #videos)