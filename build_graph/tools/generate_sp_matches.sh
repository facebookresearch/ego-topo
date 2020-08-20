dset=$1
split=$2
match_dir="build_graph/data/$dset/matches/$split"

if [ "$dset" = "gtea" ]; then
	chunks=(train1 train2 train3 train4 val)
elif [ "$dset" = "epic" ]; then
	chunks=(P01 P02 P03 P04 P05 P06 P07 P08 P10 P12 P13 P14 P15 P16 P17 P19 P20 P21 P22a P22b P23 P24 P25 P26 P27 P28 P29 P30 P31)
fi

for chunk in "${chunks[@]}"; do
	if [ ! -f "$match_dir/$chunk.pth" ]; then
		python -m build_graph.tools.generate_sp_matches --dset $dset --split $split --chunk $chunk --cv_dir build_dir/cv/tmp/superpoint/matches/$split/$chunk 
	fi
done

