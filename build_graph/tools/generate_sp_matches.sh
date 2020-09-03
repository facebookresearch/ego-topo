# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

DSET=$1
SPLIT=$2
MATCHDIR="build_graph/data/$DSET/matches/$SPLIT"

if [ "$DSET" = "gtea" ]; then
	CHUNKS=(train1 train2 train3 train4 val)
elif [ "$DSET" = "epic" ]; then
	CHUNKS=(P01 P02 P03 P04 P05 P06 P07 P08 P10 P12 P13 P14 P15 P16 P17 P19 P20 P21 P22a P22b P23 P24 P25 P26 P27 P28 P29 P30 P31)
fi

for CHUNK in "${CHUNKS[@]}"; do
	if [ ! -f "$MATCHDIR/$CHUNK.pth" ]; then
		python -m build_graph.tools.generate_sp_matches --dset $DSET --split $SPLIT --chunk $CHUNK --cv_dir build_dir/cv/tmp/superpoint/matches/$SPLIT/$CHUNK 
	fi
done

