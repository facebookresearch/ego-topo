dset=$1
N=50
for chunk in $(seq 0 $N); do
	python -m build_graph.tools.generate_sp_descriptors --chunk $chunk --nchunks $N --dset $dset --cv_dir cv/tmp/superpoint/descriptors/${dset}_${chunk}
done