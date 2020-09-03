### EPIC KITCHENS 2018
Download frames from: https://epic-kitchens.github.io/2019
Clone annotation repo from: https://github.com/epic-kitchens/epic-kitchens-55-annotations
Symlink frames and annotations to project folder.
```
mkdir -p data/epic/
ln -s /path/to/EPIC_KITCHENS_2018/epic-kitchens-55-annotations/ data/epic/annotations
ln -s /path/to/EPIC_KITCHENS_2018/frames_rgb_flow/rgb data/epic/frames/
```


### EGTEA+
Download raw videos and annotations from: http://cbs.ic.gatech.edu/fpv/
For each video, extract frames using ffmpeg:
```
for vid in videos/*.mp4; do
	dir=`basename ${vid%.*}`
	mkdir -p frames/$dir
	ffmpeg -i $vid -qscale:v 2 -vf scale=-2:256 frames/$dir/frame_%010d.jpg
done
```

Symlink frames and annotations to project folder.
```
mkdir -p data/gtea/
ln -s /path/to/EGTEA+/action-annotation/ data/gtea/annotations
ln -s /path/to/EGTEA+/frames/rgb data/gtea/frames/
```
