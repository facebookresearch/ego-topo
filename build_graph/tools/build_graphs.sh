# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Build all graphs using precomputed features/scores
# Best to submit all jobs in parallel

DSET=$1
while read VID; do
	python -m build_graph.build --dset $DSET --v_id $VID --cv_dir build_graph/cv/tmp/build_graph/$DSET/$VID
done < build_graph/data/$DSET/videos.txt