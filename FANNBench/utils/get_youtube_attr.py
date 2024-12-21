import socket
import tensorflow as tf
import os
from defination import fvecs_write, fvecs_read
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
import json
import numpy as np
import requests
import re
import concurrent.futures
from googleapiclient.errors import HttpError


def get_element_by_id(id, timeout=2):
    """
    Access a webpage and extract information using an element ID.
    """
    try:
      url = f'http://data.yt8m.org/2/j/i/{id[:2]}/{id}.js'
      # set the user agent to avoid 403 forbidden
      headers = {
          'User-Agent': 'Mozilla/5.0'
      }
      # set max 
      response = requests.get(url, headers=headers, timeout=timeout)
      # response = requests.get(url)
      if response.status_code == 200:
          pattern = re.compile(r'i\(".*?","(.*?)"\);')
          matches = pattern.findall(response.text)
          if matches:
              return matches[0]
          else:
            #   print("No matches found.")
              return None
      else:
        #   print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
          return None
      
    except requests.Timeout:
      print(f"Timeout occurred for {url}")
      return None
    except requests.RequestException as e:
      print(f"Error fetching {url}: {e}")
      return None
    
class YouTubeAPIKeyManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_key_index = 0

    def get_current_key(self):
        return self.api_keys[self.current_key_index]

    def get_cur_key_idx(self):
        return self.api_keys[self.current_key_index], self.current_key_index

    def switch_to_next_key(self, idx):
        self.current_key_index = idx + 1
        if self.current_key_index >= len(self.api_keys):
            return False
        return True

    def out_of_key(self):
        return self.current_key_index >= len(self.api_keys)


def parse_tfrecord(example_proto):
    # Define the feature description dictionary
    feature_description = {
        'id': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.VarLenFeature(tf.int64),
        'mean_rgb': tf.io.FixedLenFeature([1024], tf.float32),   # for video-level
        'mean_audio': tf.io.FixedLenFeature([128], tf.float32)   # for video-level
    }
    return tf.io.parse_single_example(example_proto, feature_description)

def get_file_list(dir, prefix):
    files = []
    for file in os.listdir(dir):
        if file.endswith(".tfrecord") and file.startswith(prefix):
            files.append(dir + file)
    return files


def get_comment_count(api_key_manager, id, video_id, index, file):
    """
    Retrieves the comment count for a YouTube video.
    """
    if api_key_manager.out_of_key():
        record = records(valid=False, valid_key=False)
        return record
    if not video_id:
        print("Invalid YouTube URL.")
        return records(valid=False, valid_key=True)
    
    api_key, key_idx = api_key_manager.get_cur_key_idx()
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    try:
        request = youtube.videos().list(
            part='statistics,snippet',
            id=video_id
        )
        response = request.execute()
        if 'items' in response and len(response['items']) > 0:
            statistics = response['items'][0]['statistics']
            comment_count = statistics.get('commentCount', 0)
            view_count = statistics.get('viewCount', 0)
            like_count = statistics.get('likeCount', 0)

            snippet = response['items'][0]['snippet']
            publish_date = snippet.get('publishedAt', None)
            record = records(id=id, real_id=video_id, idx=index, comment=int(comment_count), view=int(view_count), like=int(like_count), date=publish_date, file=file)
            return record
        else:
            # print("Video not found or is private.")
            record = records(valid=False, valid_key=True)
            return record
        
        # record = records(video_id, 1, 1, 222, 22, "101010", file) 
        # return record
    except HttpError as e:
        if e.resp.status == 403:
            error_message = e.content.decode('utf-8')
            if 'quota' in error_message:
                print(f"API key {api_key} has run out of quota.")
            else:
                print(f"API key {api_key} is invalid or access is forbidden.")
            print("switching to next key")
            api_key_manager.switch_to_next_key(key_idx)
            return get_comment_count(api_key_manager, id, video_id, index, file) # keep retrying with the next key
        elif e.resp.status == 408:
            print(f"Request timed out for video ID {video_id}. Skipping.")
            return records(valid=False, valid_key=True)
        else:
            print(f"An HTTP error occurred: {e}")
            return records(valid=False, valid_key=True)
    except socket.timeout:
        print(f"Socket timed out for video ID {video_id}. Skipping.")
        return records(valid=False, valid_key=True)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit()


class records:
    def __init__(self, id='x', real_id='x', idx=0, comment=0, view=0, like=0, date=0, file='x', valid=True, valid_key=True):
        self.id = id
        self.real_id = real_id
        self.idx = idx
        self.comment = comment
        self.view = view
        self.like = like
        self.date = date
        self.file = file
        self.valid = valid
        self.valid_key = valid_key

def get_attr_parallel(api_key_manager, idx, data, file):
    if api_key_manager.out_of_key():
        record = records(False, True)
        return record

    id = data['id'].numpy().decode('utf-8')
    # id to url
    real_id = get_element_by_id(id)
    if real_id is None:
        record = records(False, True)
        return record
    #   print("video_id: ", id, " real_id: ", real_id)
    # valid = True
    # infos = [0,0,0,0,True]
    record = get_comment_count(api_key_manager, id, real_id, idx, file)
    return record



data_size = 1200000
# query_size = 10
# train_size = 10
total_size = data_size

output_rgb_file = "/mnt/data/mocheng/dataset/youtube/data_rgb.fvecs"
output_audio_file = "/mnt/data/mocheng/dataset/youtube/data_audio.fvecs"
output_comments_file = "/mnt/data/mocheng/dataset/youtube/data_comments.json"
output_views_file = "/mnt/data/mocheng/dataset/youtube/data_views.json"
output_likes_file = "/mnt/data/mocheng/dataset/youtube/data_likes.json"
output_publish_dates_file = "/mnt/data/mocheng/dataset/youtube/data_publish_dates.json"
output_info_file = "/mnt/data/mocheng/dataset/youtube/info.json"
progress_file = "/mnt/data/mocheng/dataset/youtube/progress.json"

# load output_comments_file to check the progress

index = 1
total_cnt = 0
for i in range(1, 1000):
    comment_subfile = output_comments_file + '.' + str(i)
    if not os.path.isfile(comment_subfile):
        index = i
        print("output file index: ", index)
        break
    else:
        with open(comment_subfile, 'r') as f:
            _comments = json.load(f)
            total_cnt += len(_comments)
            print("data size: ", len(_comments), " from index ", i)
print("total size: ", total_cnt)


output_rgb_file = output_rgb_file + '.' + str(index)
output_audio_file = output_audio_file + '.' + str(index)
output_comments_file = output_comments_file + '.' + str(index)
output_views_file = output_views_file + '.' + str(index)
output_likes_file = output_likes_file + '.' + str(index)
output_publish_dates_file = output_publish_dates_file + '.' + str(index)

# youtube key
key1 = 'AIzaSyC7zEJzQyhg5BmKpIMQzo_VNq-f0nQ7CP4'
key2 = 'AIzaSyBCJKs2cfDAvRjY4vJ1Rk2eM6KWzHO-shU'
key3 = 'AIzaSyBxNBPCjWyg8nnQbXpOiVUJcRbs4lc5BpY'
key4 = 'AIzaSyCXhimXqB0c4hCc2yCoiZOEM5skpilba2Q'
key5 = 'AIzaSyAc3H2lHMd9SjZk4SUaJmeuLhg0sw7ZIUA'
key6 = 'AIzaSyBjd-UyU9vON04jNcPZ9W3MFyRKmNKecL0'
key7 = 'AIzaSyCCrwuDuw_XrHiZ6ZJBfDt9zRn7FpFNAYM'
key8 = 'AIzaSyBMCFg1M7tgiNdg7_T_EseEsb587TX6FZ8'
key9 = 'AIzaSyBnL9WEcknR5-WVnFEZO1_h5wQqg2KyWoE' 
key10 = 'AIzaSyB9PlbLyapUJayXYuAe08EfxmKF3gdr9FE'
key11 = 'AIzaSyAzwAz2P4R3pzcWop4PNKAKkqqAvYj5g9E'
key12 = 'AIzaSyDxYZFFaT27MJsWNCWQJTeHMDRhG3MMhlw'
key13 = 'AIzaSyD5WyT0bVXCO-D5XStShiVcmxIv1rL0VRI'
key14 = 'AIzaSyDH-fHAnor3l_QQQvPa0IjAr9RwBwTjnts'

api_keys = [key1, key2, key3, key4, key5, key6, key7, key8, key9, key10, key11, key12, key13, key14]  # Replace with your actual API keys
api_key_manager = YouTubeAPIKeyManager(api_keys)
# Load a dataset from TFRecord files
dir = "/mnt/data/mocheng/dataset/youtube-8m/data/yt8m/video/"
# file = "/mnt/data/mocheng/dataset/youtube-8m/data/yt8m/video/train0702.tfrecord"

# load files
files = get_file_list(dir, "train")


# wrap mean_rgb into numpy array
mean_rgb = []
mean_audio = []
comment_counts = []       
view_counts = []
like_counts = []
publish_dates = []
loaded_infos = [] # cannot match all attr because of my wrong code, now fixed

show_cnt = 0
show = 1

read_file = []
file_cnts = []


# load file if progress file exists
if os.path.isfile(progress_file):
    print("load progress file")
    with open(progress_file, 'r') as f:
        progress = json.load(f)
        read_file = progress["read_file"]
        file_cnts = progress["file_cnts"]
    # load data
    # mean_rgb = fvecs_read(output_rgb_file).tolist()
    # print("load rgb data size: ", len(mean_rgb))
    # mean_audio = fvecs_read(output_audio_file).tolist()
    # print("load audio data size: ", len(mean_audio))
    # with open(output_comments_file, 'r') as f:
    #     comment_counts = json.load(f)
    #     print("load comments data size: ", len(comment_counts))
    # with open(output_views_file, 'r') as f:
    #     view_counts = json.load(f)
    #     print("load views data size: ", len(view_counts))
    # with open(output_likes_file, 'r') as f:
    #     like_counts = json.load(f)
    #     print("load likes data size: ", len(like_counts))
    # with open(output_publish_dates_file, 'r') as f:
    #     publish_dates = json.load(f)
    #     print("load publish dates data size: ", len(publish_dates))
    # with open(output_info_file, 'r') as f:
    #     loaded_infos = json.load(f)
    #     print("load data from progress file, size=", len(mean_rgb))

    # print("load data from progress file, size=", len(mean_rgb))

            
    # mean_rgb = [mean_rgb[i] for i in valid_idx]
    # mean_audio = [mean_audio[i] for i in valid_idx]
    # comment_counts = [comment_counts[i] for i in valid_idx]
    # view_counts = [view_counts[i] for i in valid_idx]
    # like_counts = [like_counts[i] for i in valid_idx]
    # publish_dates = [publish_dates[i] for i in valid_idx]
    # loaded_infos = [loaded_infos[i] for i in valid_idx]

    # print("new size=", len(mean_rgb))
    # fvecs_write(output_rgb_file, np.array(mean_rgb))
    # fvecs_write(output_audio_file, np.array(mean_audio))
    # with open(output_comments_file, 'w') as f:
    #     json.dump(comment_counts, f, indent=1)
    # with open(output_views_file, 'w') as f:
    #     json.dump(view_counts, f, indent=1)
    # with open(output_likes_file, 'w') as f:
    #     json.dump(like_counts, f, indent=1)
    # with open(output_publish_dates_file, 'w') as f:
    #     json.dump(publish_dates, f, indent=1)
    # # save progress
    # with open(progress_file, 'w') as f:
    #     f.write(json.dumps({"read_file": read_file, "file_cnts": file_cnts}, indent=1)) # file cnts is useless, if a file did not finish, re load it
    # with open(output_info_file, 'w') as f:
    #     json.dump(loaded_infos, f, indent=1)

store_check_point = total_cnt
# iterate in reverse order
for idx, file in enumerate(files):
    if file in read_file:
        # print("skip file ", file)
        continue
    
    raw_dataset = tf.data.TFRecordDataset(file)
    parsed_dataset = raw_dataset.map(parse_tfrecord)
    
    dataset = list(parsed_dataset)
    if total_cnt == total_size:
        break
    
    timeout = 5

    results = [None] * len(dataset)
    with concurrent.futures.ThreadPoolExecutor() as executor:
      attrs = {executor.submit(get_attr_parallel, api_key_manager, idx2, data, file): idx2 for idx2, data in enumerate(dataset)}
      for future in concurrent.futures.as_completed(attrs):
        try:
            index = attrs[future]
            result = future.result()
            if(result.valid and result.valid_key):
                # print(f"Got data suc for index {index}")
                results[index] = result
            if not result.valid_key:
                print(f"Key ran out. Terminating other threads due to failure at index {index}. current size: {total_cnt}")
                results[index] = None
                exit()
                # don't need to save for incomplete data
            
        except Exception as e:
            print(f"Error processing at index {index}: {e}")
            exit()
            # results[index] = result
    
    valid_result = []
    for idx3, result in enumerate(results):
        if result is None:
            continue
        if result.valid and result.valid_key:
            valid_result.append(result)
    print("got data size:", len(valid_result), " valid percentage: ", len(valid_result)/len(dataset))

    if(total_size <= total_cnt+len(valid_result)):
        valid_result = valid_result[: total_size - total_cnt]
    
    for result in valid_result:
        idx = result.idx
        rgb = dataset[idx]['mean_rgb'].numpy()
        audio = dataset[idx]['mean_audio'].numpy()
        mean_rgb.append(rgb)
        mean_audio.append(audio)
        loaded_infos.append([result.id, result.real_id, result.idx, result.file])
        comment_counts.append(result.comment)
        view_counts.append(result.view)
        like_counts.append(result.like)
        publish_dates.append(result.date)
        total_cnt += 1

    
    if not api_key_manager.out_of_key():
        read_file.append(file)
        file_cnts.append(len(valid_result))

    # save data and current progress when each file done, to ensure security
    # if(total_cnt >= store_check_point + 1000 or api_key_manager.out_of_key()):
    #     store_check_point = total_cnt
    #     # save data
    #     fvecs_write(output_rgb_file, np.array(mean_rgb))
    #     fvecs_write(output_audio_file, np.array(mean_audio))
    #     with open(output_comments_file, 'w') as f:
    #         json.dump(comment_counts, f, indent=1)
    #     with open(output_views_file, 'w') as f:
    #         json.dump(view_counts, f, indent=1)
    #     with open(output_likes_file, 'w') as f:
    #         json.dump(like_counts, f, indent=1)
    #     with open(output_publish_dates_file, 'w') as f:
    #         json.dump(publish_dates, f, indent=1)
    #     with open(output_info_file, 'w') as f:
    #         json.dump(loaded_infos, f, indent=1)
    #     # save progress
    #     with open(progress_file, 'w') as f:
    #         f.write(json.dumps({"read_file": read_file, "file_cnts": file_cnts}, indent=1))
    #     print("save data and current progress")
    print("save data and current progress")
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
    # save progress
    with open(progress_file, 'w') as f:
        f.write(json.dumps({"read_file": read_file, "file_cnts": file_cnts}, indent=1)) # file cnts is useless, if a file did not finish, re load it
    with open(output_info_file, 'w') as f:
        json.dump(loaded_infos, f, indent=1)

    print("process done for file ", file, " got size ", total_cnt)
            
    if(total_cnt >= total_size):
        print("task done for ", total_size, " items")
        exit()
    if(api_key_manager.out_of_key()):
        print("key out of usage, ", len(mean_rgb), " stored to index: ", index, " total size: ", total_cnt)
        exit()


