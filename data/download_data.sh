# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Download all precomputed data and models from the public S3 bucket

BUCKET=https://dl.fbaipublicfiles.com/ego-topo
function dls3 {
	mkdir -p `dirname $1`
	wget -O $1 $BUCKET/$1
}

function dls3_unzip {
	mkdir -p `dirname $1`
	wget -O $1 $BUCKET/$1
	unzip $1 -d `dirname $1`
	rm $1
}

# download dataset metadata -- train/val splits etc. (628K + 238K)
# Required for all experiments
dls3_unzip data/epic/splits.zip
dls3_unzip data/gtea/splits.zip
echo 'Downloaded dataset splits'

# download precomputed neighbor files (148M + 388M)
# Required to train localization network
dls3_unzip build_graph/data/epic/neighbor_data.zip
dls3_unzip build_graph/data/gtea/neighbor_data.zip
echo 'Downloaded precomputed neighbor files'

# download localization network checkpoints (136M + 136M)
# Required to generate graph
dls3 build_graph/localization_network/cv/epic/ckpt_E_250_I_19250_L_0.368_A_0.834.pth
dls3 build_graph/localization_network/cv/gtea/ckpt_E_250_I_6750_L_0.281_A_0.913.pth
echo 'Downloaded localization network checkpoints'

# download superpoint checkpoint (5M)
# Required to generate homography matches
dls3 build_graph/tools/superpoint/superpoint_v1.pth
echo 'Downloaded pretrained superpoint model'

# download precomputed graphs (550M + 629M)
dls3_unzip build_graph/data/epic/graphs.zip
dls3_unzip build_graph/data/gtea/graphs.zip
echo 'Downloaded precomputed graphs'

# download clip features for anticipation experiments (2.3G + 1.9G)
# Required to run anticipation models
dls3_unzip anticipation/data/epic/features.zip
dls3_unzip anticipation/data/gtea/features.zip
echo 'Downloaded precomputed I3D clip features for anticipation'

# download pretrained anticipation models (2.9G)
dls3_unzip anticipation/pretrained.zip
echo 'Downloaded pretrained anticipation models'
