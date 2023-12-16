import requests
import sys
import time
import winsound
import os
import re
import argparse
import json
import glob
import threading
import queue
from chatbot import bot

temp_path = f"{os.path.dirname(__file__)}/../Temp"

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', default='true')
parser.add_argument('--voice', action='store_true')
parser.add_argument('--game', action='store_true')
parser.add_argument('--load_memory', action='store_true')

args = parser.parse_args()

textQ = queue.Queue()
text_to_voiceQ = queue.Queue()
voiceQ = queue.Queue()

def generate_text():

  character_template_path = f"{os.path.dirname(__file__)}/character_template/1.txt"
  chat = bot(
    character_template_path=character_template_path,
    load_memory=args.load_memory
  )
  
  blank = "前の話題に具体性を持たせ、独り言をしてください。もしくは、新たな話題で独り言をしてください。"

  if args.game == True:
    while True:
      if os.path.isfile(f"{temp_path}/orc.txt"):
        with  open(f"{temp_path}/orc.txt", 'r', encoding='UTF-8', newline='\n') as f:
          input_message = f.read()

        text_to_Q(chat, blank, input_message)

  else:
    while True:

      if textQ.qsize() > 20:
        textQ.queue.clear()

        if args.debug == False:
          with open(f"{temp_path}/comment.txt", 'w', encoding='UTF-8') as f:
            f.write("")

      if args.debug:
        input_message = input("あなた: ")

        if input_message.isspace() or input_message == "":
          input_message = blank

      else:
        input_message = ""

        if os.path.isfile(f"{temp_path}/comment.txt"):
          with open(f"{temp_path}/comment.txt", 'r', encoding='UTF-8', newline='\n') as f:
            lines = f.readlines()

          if lines: # linesが空でなければ
            input_message = lines[0].strip()  
            new_lines = [line for line in lines if line.strip() != input_message]

            with open(f"{temp_path}/comment.txt", 'w', encoding='UTF-8', newline='\n') as f:
              f.writelines(new_lines)

          if input_message.isspace() or input_message == "":
            input_message = blank

      text_to_Q(chat, blank, input_message)


def text_to_Q(chat, blank,input_message):
  textQ.put(input_message)
  
  if not (input_message == blank or input_message.isspace() ):
    text_to_voiceQ.put("【入力】"+input_message)

  textQ_to_voiceQ(chat)


def textQ_to_voiceQ(chat):
  if not os.path.isfile(f"{temp_path}/thinking"):
    f = open(f"{temp_path}/thinking", 'w')
    f.close()

  RealTimeResponce = ""
  target_char = ["。", "！", "？", "\n"]

  input_message = textQ.get()

  print("AI: ", end="")
  for chunk in chat.reply(input_message):
    sys.stdout.write(chunk)
    sys.stdout.flush()
    RealTimeResponce += chunk

    for char in target_char:
      if char in RealTimeResponce: 
        text_to_voiceQ.put(RealTimeResponce)       
        RealTimeResponce = ""

  if RealTimeResponce:
    text_to_voiceQ.put(RealTimeResponce)

  print("\n")


def text_to_voice():
  tag = "【平常】"
  while True:
    if  text_to_voiceQ.qsize() > 0:
      ai_response = text_to_voiceQ.get()

      # 【】で囲まれた部分を抽出
      matches = re.findall(r"【(.*?)】", ai_response)
      if matches:
        tag = "【" + matches[0] + "】"
        text = re.sub(r"【.*?】", "", ai_response)

      else:
        text = ai_response

      if tag == "【喜び】":
        speaker = 24 #たのしい
      elif tag == "【悲しみ】":
        speaker = 25 #かなしい
      elif tag == "【恐怖】":
        speaker = 26 #ぴえーん
      else:
        speaker = 23 #ノーマル

      # 音声合成クエリの作成
      res1 = requests.post('http://127.0.0.1:50021/audio_query',params = {'text': text, 'speaker': speaker})
      # 音声合成データの作成
      res2 = requests.post('http://127.0.0.1:50021/synthesis',params = {'speaker': speaker},data=json.dumps(res1.json()))

      # wavデータの生成
      file_name = f"{temp_path}/{str(time.time())}.wav"
      with open(file_name, mode='wb') as f:
        f.write(res2.content)
      voiceQ.put(file_name)

      if tag == "【入力】":
        with open(f"{temp_path}/input_message.txt", 'w', encoding='UTF-8', newline='\n') as f:
          f.write(text + '\n')


def speak():
  while True: 
    if  voiceQ.qsize() > 0:

      if os.path.isfile(f"{temp_path}/thinking"):
        os.remove(f"{temp_path}/thinking")  

      file_name = voiceQ.get()
      winsound.PlaySound(file_name, winsound.SND_FILENAME)
      os.remove(file_name)

    if os.path.isfile(f"{temp_path}/orc.txt") and textQ.qsize() == 0 and len(glob.glob('{temp_path}/*.wav')) == 0:
      os.remove(f"{temp_path}/orc.txt")

    if os.path.isfile(f"{temp_path}/input_message.txt") and voiceQ.qsize() == 0 and len(glob.glob('{temp_path}/*.wav')) == 0:
      os.remove(f"{temp_path}/input_message.txt")


def get_text_from_game():
  import orc
  while True:
    if not os.path.isfile(f"{temp_path}/orc.txt"):
      orc.press_enter()
      orc.get_text()


def run():
  print("Talk.py is running")
  
  thread_run = threading.Thread(target=generate_text,daemon=True)
  thread_run.start()

  if args.voice == True:
    thread_generate_VOICEVOX = threading.Thread(target=text_to_voice, daemon=True)
    thread_generate_VOICEVOX.start()
    
  if args.game == True:
    thread_game_mode = threading.Thread(target=get_text_from_game, daemon=True)
    thread_game_mode.start()

  speak()

if __name__ == "__main__":
  run()