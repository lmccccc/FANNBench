import numpy as np
import os
from defination import fvecs_read, fvecs_write
import json


output_rgb_file = "/mnt/data/mocheng/dataset/youtube/data_rgb.fvecs"
output_audio_file = "/mnt/data/mocheng/dataset/youtube/data_audio.fvecs"
output_comments_file = "/mnt/data/mocheng/dataset/youtube/data_comments.json"
output_views_file = "/mnt/data/mocheng/dataset/youtube/data_views.json"
output_likes_file = "/mnt/data/mocheng/dataset/youtube/data_likes.json"
output_publish_dates_file = "/mnt/data/mocheng/dataset/youtube/data_publish_dates.json"
progress_file = "/mnt/data/mocheng/dataset/youtube/progress.json"

index = 1
for i in range(1, 1000):
    comment_subfile = output_comments_file + '.' + str(i)
    if not os.path.isfile(comment_subfile):
        index = i
        break

print("index from 1 to ", index-1)


mean_rgb = []
mean_audio = []
comment_counts = []       
view_counts = []
like_counts = []
publish_dates = []
loaded_infos = [] # cannot match all attr because of my wrong code, now fixed


for i in range(1, index):
    rgb_file = output_rgb_file + '.' + str(i)
    audio_file = output_audio_file + '.' + str(i)
    comments_file = output_comments_file + '.' + str(i)
    views_file = output_views_file + '.' + str(i)
    likes_file = output_likes_file + '.' + str(i)
    publish_dates_file = output_publish_dates_file + '.' + str(i)

    if not os.path.isfile(rgb_file):
        print("file not exist: ", rgb_file)
        exit()

    t_mean_rgb = fvecs_read(rgb_file).tolist()
    t_mean_audio = fvecs_read(audio_file).tolist()
    assert(len(t_mean_rgb) == len(t_mean_audio))
    with open(comments_file, 'r') as f:
        t_comment_counts = json.load(f)
        assert(len(t_mean_rgb) == len(t_comment_counts))
    with open(views_file, 'r') as f:
        t_view_counts = json.load(f)
        assert(len(t_mean_rgb) == len(t_view_counts))
    with open(likes_file, 'r') as f:
        t_like_counts = json.load(f)
        assert(len(t_mean_rgb) == len(t_like_counts))
    with open(publish_dates_file, 'r') as f:
        t_publish_dates = json.load(f)
        assert(len(t_mean_rgb) == len(t_publish_dates))

    print("load data from progress file, size=", len(t_mean_rgb))
    
    mean_rgb += t_mean_rgb
    mean_audio += t_mean_audio
    comment_counts += t_comment_counts
    view_counts += t_view_counts
    like_counts += t_like_counts
    publish_dates += t_publish_dates

print("total size=", len(mean_rgb))
# save data
fvecs_write(output_rgb_file, np.array(mean_rgb))
fvecs_write(output_audio_file, np.array(mean_audio))
with open(output_comments_file, 'w') as f:
    json.dump(comment_counts, f, indent=1)
with open(output_views_file, 'w') as f:
    json.dump(view_counts, f, indent=1)
with open(output_likes_file, 'w') as f:
    json.dump(like_counts, f, indent=1)
with open(output_publish_dates_file, 'w') as f:
    json.dump(publish_dates, f, indent=1)

print("merge done")