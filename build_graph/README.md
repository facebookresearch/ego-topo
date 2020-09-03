# Graph building

Follow the instructions in the root level `README.md` to download the datasets and precomputed features.

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

**Train a localization network**. See `build_graph/localization_network/README.md` for full details.
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