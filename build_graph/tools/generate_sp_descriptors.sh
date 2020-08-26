DSET=$1
N=50
for CHUNK in $(seq 0 $N); do
	python -m build_graph.tools.generate_sp_descriptors --chunk $CHUNK --nchunks $N --dset $DSET --cv_dir cv/tmp/superpoint/descriptors/${DSET}_${CHUNK}
done