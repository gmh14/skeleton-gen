import os, json, sys

cmd = "/home/mhg/Projects/sga-2024/libpgo/build/bin/skeletonViewer {} /home/mhg/Projects/skeleton-gen/all_data/{}.json"

folders = ["/media/mhg/data/Dropbox/medial-axis/Results/resultsAll/",
           "/media/mhg/data/Dropbox/medial-axis/Results/results/",
           "/media/mhg/data/Dropbox/medial-axis/Results/paper-demo/",
           "/media/mhg/data/Dropbox/medial-axis/Results/paper-demo-final/"]

for folder in folders:
    for dir in os.listdir(folder):
        if not os.path.isdir(f"{folder}/{dir}"):
            continue
        
        skl_file = f"{folder}/{dir}/opt.ma.post.json"
        os.system(cmd.format(skl_file, dir))
        