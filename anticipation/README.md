# Long-term Anticipation

### Preparation
Prepare the data files for EPIC and GTEA under `data/epic/` and `data/gtea` folders, including:
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
python -m epic.tools.train_recognizer epic/configs/long_term_anti/epic/verb_anti_vpretrain/i3d.py
# GFB_GCN model
python -m epic.tools.train_recognizer epic/configs/long_term_anti/epic/verb_anti_vpretrain/gfb_gcn.py
```
