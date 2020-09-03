# Long-term Anticipation

### Preparation

Follow the instructions in the root level `README.md` to download the datasets and precomputed features.

**Generate (or download) the graph files for EPIC and GTEA** following the instructions in `build_graph/Readme.md`. Then create symlinks to the data folders.
```
mkdir -p anticipation/data/epic anticipation/data/gtea
ln -s `pwd`/data/epic/* `pwd`/build_graph/data/epic/graphs anticipation/data/epic/
ln -s `pwd`/data/gtea/* `pwd`/build_graph/data/gtea/graphs anticipation/data/gtea/
```

If followed correctly, each dataset folder should have the following folders. 
```
> ls anticipation/data/${dataset}
annotations  features  frames  graphs  split  videos.txt
```
See `data/download_data.sh` for any missing files.

**Install custom branches of libraries** provided as submodules:
* `mmaction`: following INSTALL.md in `anticipation/mmcv`.
* `mmcv`: following README.rst in `anticipation/mmaction`.


### Train anticipation model
Specify corresponding config files for training different models:
```
cd anticipation
python -m anticipation.tools.train_recognizer anticipation/configs/long_term_anti/{epic/gtea}/verb_anti_vpretrain/{method}.py
```

For the final results, "mAP_0", "mAP_1" and "mAP_2" corresponds to the mAP values for all action classes, many-shot (freq) classes and low-shot (rare) classes, respectivaly. 


### Test trained anticpation model
After training the models or directly downloading our pretrained models, you can test with:
```
cd anticipation
python -m anticipation.tools.test_recognizer anticipation/configs/long_term_anti/${dataset}/verb_anti_vpretrain/${method}.py \
    --checkpoint PATH_TO_MODELS/latest.pth
```

For example, our pretrained GCN model on EPIC:
```
> python -m anticipation.tools.test_recognizer anticipation/configs/long_term_anti/epic/verb_anti_vpretrain/gfb_gcn.py --checkpoint pretrained/epic/gfb_gcn/latest.pth

{'mAP_0': 0.379436194896698,
 'mAP_1': 0.5639347434043884,
 'mAP_2': 0.29221880435943604,
 'mAP_ratio_0': 0.4598943591117859,
 'mAP_ratio_1': 0.430255264043808,
 'mAP_ratio_2': 0.35407784581184387}
```

Note that the results may not be exactly the same as the reported results in the paper as we averaged the results over 5 runs in the paper.
