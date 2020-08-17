# Egocentric-Topological-Maps

### Training
```
python -m epic.tools.train_recognizer epic/configs/baseline/fixfeature_S1_2mlp.py
```

### Generate LFB features
```
stride=30
python -m epic.tools.generate_lfb  epic/configs/finetune/i3d_r50_S1_f32s2_kpretrain_fixbn.py \
    data/epic/split/train_S1_nframes.csv data/epic/split/train_S1_verb.csv \
    --lfb_output train_lfb_s${stride}.pkl --lfb_class_output train_lfb_class_s${stride}.pkl  --lfb_clip_stride ${stride}
```
