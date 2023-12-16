import pytchat
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video_id', default='string')
args = parser.parse_args()

temp_path = f"{os.path.dirname(__file__)}/../Temp"

# PytchatCoreオブジェクトの取得
# video_idはhttps://....watch?v=より後ろのやつ
livechat = pytchat.create(video_id = args.video_id)

while livechat.is_alive():

  chatdata = livechat.get()
  for c in chatdata.items:
    if os.path.isfile(f"{temp_path}/comment.txt"):
      f = open(f"{temp_path}/comment.txt", 'a', encoding='UTF-8', newline='\n')
    else:
      f = open(f"{temp_path}/comment.txt", 'w', encoding='UTF-8', newline='\n')

    f.write(c.message + "\n")
    f.close()
  time.sleep(4)