import numpy as np
import os
import sys

from defination import fvecs_read, fvecs_write
import json
import random
from datetime import datetime

data_size  = 1000000
train_size = 100000
query_size = 10000

root = "/mnt/data/mocheng/dataset/youtube/"
output_audio_root = "/mnt/data/mocheng/dataset/youtube_audio1m/"
output_root = "/mnt/data/mocheng/dataset/youtube1m/"



input_rgb_file = root + "data_rgb.fvecs"
input_audio_file = root + "data_audio.fvecs"
input_comments_file = root + "data_comments.json"
input_views_file = root + "data_views.json"
input_likes_file = root + "data_likes.json"
input_publish_dates_file = root + "data_publish_dates.json"
progress_file = root + "progress.json"

output_rgb = output_root + "rgb.fvecs"
output_rgb_train = output_root + "rgb_train.fvecs"
output_rgb_query = output_root + "rgb_query.fvecs"

output_audio = output_audio_root + "audio.fvecs"
output_audio_train = output_audio_root + "audio_train.fvecs"
output_audio_query = output_audio_root + "audio_query.fvecs"

output_likes = output_root + "likes.json"
output_likes_query = output_root + "likes_query.json"

output_views = output_root + "views.json"
output_views_query = output_root + "views_query.json"

output_comments = output_root + "comments.json"
output_comments_query = output_root + "comments_query.json"

output_dates = output_root + "dates.json"
output_dates_query = output_root + "dates_query.json"

index = 1
for i in range(1, 1000):
    comment_subfile = input_comments_file + '.' + str(i)
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

max_int = (1 << 31) - 1
total_remove = 0
for i in range(1, index):
    rgb_file = input_rgb_file + '.' + str(i)
    audio_file = input_audio_file + '.' + str(i)
    comments_file = input_comments_file + '.' + str(i)
    views_file = input_views_file + '.' + str(i)
    likes_file = input_likes_file + '.' + str(i)
    publish_dates_file = input_publish_dates_file + '.' + str(i)

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
    
    cur_remove = 0
    for i in range(len(t_mean_rgb)):
        if t_view_counts[i] == 0:
            cur_remove += 1
            continue
        mean_rgb.append(t_mean_rgb[i])
        mean_audio.append(t_mean_audio[i])
        if t_comment_counts[i] > max_int:
            t_comment_counts[i] = max_int
        if t_view_counts[i] > max_int:
            t_view_counts[i] = max_int
        if t_like_counts[i] > max_int:
            t_like_counts[i] = max_int
        comment_counts.append(t_comment_counts[i])
        view_counts.append(t_view_counts[i])
        like_counts.append(t_like_counts[i])
        publish_dates.append(t_publish_dates[i])
    total_remove += cur_remove
    print("remove ", cur_remove, " data, total remove ", total_remove)
    # mean_rgb += t_mean_rgb
    # mean_audio += t_mean_audio
    # comment_counts += t_comment_counts
    # view_counts += t_view_counts
    # like_counts += t_like_counts
    # publish_dates += t_publish_dates

print("total size=", len(mean_rgb))
indices = list(range(len(mean_rgb)))
# Shuffle the indices
random.shuffle(indices)
mean_rgb = [mean_rgb[i] for i in indices]
mean_audio = [mean_audio[i] for i in indices]
comment_counts = [comment_counts[i] for i in indices]
view_counts = [view_counts[i] for i in indices]
like_counts = [like_counts[i] for i in indices]
publish_dates = [publish_dates[i] for i in indices]

publish_dates_encoded = []
for date in publish_dates:
    # Parse the date string into a datetime object
    if(date == 0 or date == "0"):
        publish_dates_encoded.append(0)
        continue
    date_object = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ")
    # Convert the datetime object to a timestamp (seconds since epoch)
    timestamp = int(date_object.timestamp())
    publish_dates_encoded.append(timestamp)


# save data
index = [0, data_size, data_size+query_size, data_size+train_size+query_size]
fvecs_write(output_rgb, np.array(mean_rgb[index[0]: index[1]]))
print("save rgb to ", output_rgb)
fvecs_write(output_rgb_query, np.array(mean_rgb[index[1]: index[2]]))
print("save rgb query to ", output_rgb_query)
fvecs_write(output_rgb_train, np.array(mean_rgb[index[2]: index[3]]))
print("save rgb train to ", output_rgb_train)

fvecs_write(output_audio, np.array(mean_audio[index[0]: index[1]]))
print("save audio to ", output_audio)
fvecs_write(output_audio_query, np.array(mean_audio[index[1]: index[2]]))
print("save audio query to ", output_audio_query)
fvecs_write(output_audio_train, np.array(mean_audio[index[2]: index[3]]))
print("save audio train to ", output_audio_train)

with open(output_comments, 'w') as f:
    json.dump(comment_counts[index[0]: index[1]], f, indent=1)
print("save comments to ", output_comments)
with open(output_comments_query, 'w') as f:
    json.dump(comment_counts[index[1]: index[2]], f, indent=1)
print("save comments query to ", output_comments_query)

with open(output_views, 'w') as f:
    json.dump(view_counts[index[0]: index[1]], f, indent=1)
print("save views to ", output_views)
with open(output_views_query, 'w') as f:
    json.dump(view_counts[index[1]: index[2]], f, indent=1)
print("save views query to ", output_views_query)
    
with open(output_likes, 'w') as f:
    json.dump(like_counts[index[0]: index[1]], f, indent=1)
print("save likes to ", output_likes)
with open(output_likes_query, 'w') as f:
    json.dump(like_counts[index[1]: index[2]], f, indent=1)
print("save likes query to ", output_likes_query)

with open(output_dates, 'w') as f:
    json.dump(publish_dates_encoded[index[0]: index[1]], f, indent=1)
print("save dates to ", output_dates)

with open(output_dates_query, 'w') as f:
    json.dump(publish_dates_encoded[index[1]: index[2]], f, indent=1)
print("save dates query to ", output_dates_query)

# with open(input_comments_file, 'w') as f:
#     json.dump(comment_counts, f, indent=1)
# with open(input_views_file, 'w') as f:
#     json.dump(view_counts, f, indent=1)
# with open(input_likes_file, 'w') as f:
#     json.dump(like_counts, f, indent=1)
# with open(input_publish_dates_file, 'w') as f:
#     json.dump(publish_dates, f, indent=1)
# fvecs_write(output_rgb, np.array(mean_rgb))
# fvecs_write(input_audio_file, np.array(mean_audio))

# fvecs_write(output_rgb, np.array(mean_audio))
# fvecs_write(output_rgb, np.array(mean_audio))
print("merge done")