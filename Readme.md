# EGO-TOPO: Environment Affordances from Egocentric Video

![Teaser](http://vision.cs.utexas.edu/projects/ego-topo/media/concept.png)

This code implements a model to parse untrimmed egocentric video into a topological graph of activity zones in the underlying environment, and associates actions that occur in the video to nodes in this graph. The topological graph serves as a structured representation of the video and can be used to discover environment affordances and help anticipate future actions in long form video. See the demo [here]!(http://vision.cs.utexas.edu/projects/ego-topo/demo.html)

This is the code accompanying our CVPR20 (oral) work:  
Tushar Nagarajan, Yanghao Li, Christoph Feichtenhofer and Kristen Grauman.  
EGO-TOPO: Environment Affordances from Egocentric Video [[arxiv]](https://arxiv.org/pdf/2001.04583.pdf) [[project page]](http://vision.cs.utexas.edu/projects/ego-topo/)

## Graph construction

Construct the graph for video P26_01 in the EPIC Kitchens dataset using a trained localization network and visualize the output:
```
python -m build_graph.build --dset epic --v_id P26_01 --locnet_wts build_graph/localization_network/cv/epic/ckpt_E_250_I_19250_L_0.368_A_0.834.pth --online --frame_inp --viz
```

Full instructions can be found in `build_graph/Readme.md`


## Long term action anticipation

Use the constructed graphs as a video representation to train action anticipation models:
```
cd anticipation
python -m epic.tools.train_recognizer epic/configs/long_term_anti/epic/verb_anti_vpretrain/gfb_gcn.py
```

Full instructions can be found in `anticipation/Readme.md`

## Demo

Explore the constructed graphs using our method at our [demo page](http://vision.cs.utexas.edu/projects/ego-topo/demo.html).


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