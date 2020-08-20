# Graph building

**Download the [EPIC Kitchens 55](https://epic-kitchens.github.io/2019) and [EGTEA+](http://cbs.ic.gatech.edu/fpv/) datasets** and symlink the frames and annotations directories to the respective data folder. See detailed instructions in `build_graph/data/datasets.md`.

```
mkdir -p build_graph/data/epic/
ln -s /path/to/EPIC_KITCHENS/epic-kitchens-55-annotations/ build_graph/data/epic/annotations
ln -s /path/to/EPIC_KITCHENS/frames_rgb_flow/rgb build_graph/data/epic/frames/

mkdir -p build_graph/data/gtea/
ln -s /path/to/EGTEA+/action-annotation/ build_graph/data/gtea/annotations
ln -s /path/to/EGTEA+/frames/rgb build_graph/data/gtea/frames/
```

**Download all precomputed features and pretrained models** to construct topological graphs from videos. See requirements below for details on how to compute the required features and how to train the required models.
```
bash build_graph/tools/download_precomputed_data.sh
```

**Build the graph for each video**. Use `--viz` to visualize, omit to save files. For example, for video `P26_01` in EPIC:
```
# Using just the frames from the video (very slow)
python -m build_graph.build --dset epic --v_id P26_01 --locnet_wts build_graph/localization_network/cv/epic/ckpt_E_250_I_19250_L_0.368_A_0.834.pth --online --frame_inp --viz

# Using precomputed localization net trunk features (slow). See requirements below
python -m build_graph.build --dset epic --v_id P26_01 --locnet_wts build_graph/localization_network/cv/epic/ckpt_E_250_I_19250_L_0.368_A_0.834.pth --online --viz

# Using precomputed localization net scores (fast). See requirements below
python -m build_graph.build --dset epic --v_id P26_01 --viz
```
Outputs: 
```
build_graph/data/epic/graphs/P26_01_graph.pth
```

### Requirements:

**Train a localization network**. See `build_graph/localization_network/Readme.md` for full details.
```
python -m build_graph.localization_network.train --dset epic --max_iter 20000 --cv_dir build_graph/localization_network/cv/epic/ --parallel 
```
Outputs:
```
build_graph/localization_network/cv/epic/ckpt_E_250_I_19250_L_0.368_A_0.834.pth
```

-------

**(Optional) Precompute localization network features** for all frames in the dataset to speed up graph construction. *Recommended*.
```
 python -m build_graph.tools.generate_locnet_features --load build_graph/localization_network/cv/epic/ckpt_E_250_I_19250_L_0.368_A_0.834.pth --dset epic
```
Outputs: 
```
build_graph/data/epic/locnet_trunk_feats.pth
```

-------
**(Optional) Precompute pairwise localization net similarity matrix** per video to speed up graph construction further. *Recommended*.
```
python -m build_graph.tools.generate_pairwise_locnet_scores --load build_graph/localization_network/cv/epic/head.pth --dset epic --parallel --v_id P26_01
```
Outputs:
```
build_graph/data/epic/locnet_pairwise_scores/P26_01_scores.pth
```