import json
import os
import numpy as np

filelist = ["./testing_data.json", "./train_val_data.json"]
feature_dir = ["./test_features/", "./train_features/"]


for i, f in enumerate(filelist):
    f = json.load(open(f, 'r'))
    exist_videos = []
    for video in f:
        if os.path.isfile(feature_dir[i] + video["id"] + ".audio.npy") and os.path.isfile(feature_dir[i] + video["id"] + ".vgg.npy"):
            if np.load(feature_dir[i] + video["id"] + ".audio.npy").shape == (80, 300) and np.load(feature_dir[i] + video["id"] + ".vgg.npy").shape == (80, 4096):
                exist_videos.append(video)
    json.dump(exist_videos, open(filelist[i].rstrip(".json") + "_exist.json", 'w'), indent=4)



