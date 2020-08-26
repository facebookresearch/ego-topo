# Generate pairwise localization net scores for all frames in each video
# Best to submit all jobs in parallel

DSET=$1
while read VID; do
	python -m build_graph.tools.generate_pairwise_locnet_scores --dset $DSET --v_id $VID --load build_graph/localization_network/cv/$DSET/head.pth --parallel
done < build_graph/data/$DSET/videos.txt