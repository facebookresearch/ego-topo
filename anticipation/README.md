# Long-term Anticipation

### Preparation
Prepare the data files for EPIC and GTEA datasets by following the data preparation and graph building process in `build_graph/Readme.md`. Then create symlinks to the `data` folder in `build_graph`.
```
mkdir -p data
ln -s ../../build_graph/data/epic data/epic
ln -s ../../build_graph/data/gtea data/gtea
```
Each dataset folder should contain:
```
annotations: annotations files
features: extracted clip-based features
graphs: generated Ego-Topo graphs
split: train/test split files
```

Make sure the required libraries are installed, like
* `mmaction`: following the INSTALL.md in anticipation/mmaction folder.
* `dgl`: conda install -c dglteam dgl-cuda10.0==0.4

### Train anticipation model
Specify corresponding config files for training different models:
```
python -m anticipation.tools.train_recognizer anticipation/configs/long_term_anti/{epic/gtea}/verb_anti_vpretrain/{method}.py
```

For the final results, "mAP_0", "mAP_1" and "mAP_2" corresponds to the mAP values for all action classes, many-shot (freq) classes and low-shot (rare) classes, respectivaly. 


### Test pretrained anticpation model
After training the models or directly download our pretrained models, you can test the pretrained models with:
```
python -m anticipation.tools.test_recognizer anticipation/configs/long_term_anti/${dataset}/verb_anti_vpretrain/${method}.py \
    --checkpoint PATH_TO_MODELS/latest.pth
```

You can also test all the pretrained models using script `sh test_pretrained.sh`. Note that the results may not be exactly the same as the reported results in the paper as we averaged the results over 5 runs in the paper.
