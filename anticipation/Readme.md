# Long-term Anticipation

### Preparation
Prepare the data files for EPIC and GTEA datasets by following the data preparation and graph building process in `build_graph/Readme.md`. Then create symlink to the `data` folder in `build_graph`.
```
mkdir -p data
ln -s ../../build_graph/data/epic data/epic
ln -s ../../build_graph/data/gtea data/gtea
```
Each dataset folder should contains:
```
annotations: annotations files
features: extracted clip-based features
graphs: generated Ego-Topo graphs
split: train/test split files
```

### Training anticipation moodel
Specify corresponding config files for training models:
```
# I3D model
python -m anticipation.tools.train_recognizer epic/configs/long_term_anti/epic/verb_anti_vpretrain/i3d.py
# GFB_GCN model
python -m anticipation.tools.train_recognizer epic/configs/long_term_anti/epic/verb_anti_vpretrain/gfb_gcn.py
```
