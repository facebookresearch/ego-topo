# EGO-TOPO: Environment Affordances from Egocentric Video

![Teaser](http://vision.cs.utexas.edu/projects/ego-topo/media/concept.png)

This code implements a model to parse untrimmed egocentric video into a topological graph of activity zones in the underlying environment, and associates actions that occur in the video to nodes in this graph. The topological graph serves as a structured representation of the video and can be used to discover environment affordances and help anticipate future actions in long form video. See the demo [here](http://vision.cs.utexas.edu/projects/ego-topo/demo.html)!

This is the code accompanying our CVPR20 (oral) work:  
**EGO-TOPO: Environment Affordances from Egocentric Video**  
*Tushar Nagarajan, Yanghao Li, Christoph Feichtenhofer and Kristen Grauman.*  
[[arxiv]](https://arxiv.org/pdf/2001.04583.pdf) [[project page]](http://vision.cs.utexas.edu/projects/ego-topo/)

## Requirements

Clone the repo and the associated submodules:
```
git clone --recursive https://github.com/facebookresearch/ego-topo
```

Install required packages:
```
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
```
Code tested with Python 3.7, Pytorch 1.1, cuda 10.0, cudnn 7.6

## Dataset and precomputed data files

**Download the [EPIC Kitchens 55](https://epic-kitchens.github.io/2019) and [EGTEA+](http://cbs.ic.gatech.edu/fpv/) datasets** and symlink the frames and annotations directories to the respective data folder. See detailed instructions in `build_graph/data/DATASETS.md`.

```
mkdir -p data/epic/
ln -s /path/to/EPIC_KITCHENS/epic-kitchens-55-annotations/ data/epic/annotations
ln -s /path/to/EPIC_KITCHENS/frames_rgb_flow/rgb data/epic/frames/

mkdir -p data/gtea/
ln -s /path/to/EGTEA+/action-annotation/ data/gtea/annotations
ln -s /path/to/EGTEA+/frames/rgb data/gtea/frames/
```

**Download all precomputed features and pretrained models** to construct topological graphs from videos. See requirements below for details on how to compute the required features and how to train the required models. Please see the .sh file for a list of what is downloaded.
```
bash data/download_data.sh
```

## Graph construction

Construct the graph for video P26_01 in the EPIC Kitchens dataset using a trained localization network and visualize the output:
```
# (very slow)
python -m build_graph.build --dset epic --v_id P26_01 --locnet_wts build_graph/localization_network/cv/epic/ckpt_E_250_I_19250_L_0.368_A_0.834.pth --online --frame_inp --viz

# (fast) but requires precomputing pairwise frame similarity scores. See `build_graph/README.md`
python -m build_graph.build --dset epic --v_id P26_01 --viz

```

Full instructions can be found in `build_graph/README.md`


## Long term action anticipation

Use the constructed graphs as a video representation to train action anticipation models:
```
cd anticipation
python -m anticipation.tools.train_recognizer epic/configs/long_term_anti/epic/verb_anti_vpretrain/gfb_gcn.py
```

Evaluate models using
```
python -m anticipation.tools.test_recognizer anticipation/configs/long_term_anti/epic/verb_anti_vpretrain/gfb_gcn.py --checkpoint pretrained/epic/gfb_gcn/latest.pth
```

Full instructions can be found in `anticipation/README.md`

## Demo

Explore the constructed graphs using our method at our [demo page](http://vision.cs.utexas.edu/projects/ego-topo/demo.html).


## License

This project is released under the CC-BY-NC 4.0 license, as found in the LICENSE file.


## Cite

If you find this repository useful in your own research, please consider citing:
```
@inproceedings{ego-topo,
    author = {Nagarajan, Tushar and Li, Yanghao and Feichtenhofer, Christoph and Grauman, Kristen},
    title = {EGO-TOPO: Environment Affordances from Egocentric Video},
    booktitle = {CVPR},
    year = {2020}
}
```
