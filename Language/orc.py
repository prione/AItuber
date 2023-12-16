import cv2
from PIL import Image, ImageGrab
import pyocr
import numpy as np
import winxpgui
import win32gui
import pyautogui
import time

def WindowCapture(window_name: str):
  # 現在アクティブなウィンドウ名を探す
  process_list = []

  def callback(handle, _):
    process_list.append(win32gui.GetWindowText(handle))
  win32gui.EnumWindows(callback, None)

  # ターゲットウィンドウ名を探す
  flag = True
  while flag:
    for process_name in process_list:
      if window_name in process_name:
        handle = win32gui.FindWindow(None, process_name)
        flag = False
        break
    else:
      print("ウィンドウがないよ")
      time.sleep(5)

  rect = winxpgui.GetWindowRect(handle)
  return process_name, rect

name, region = WindowCapture(input("ウィンドウ名: "))

def press_enter():
  pyautogui.getWindowsWithTitle(name)[0].activate()

  # ウィンドウがアクティブになるまで少し待つ（必要に応じて調整）
  time.sleep(0.1)

  # エンターキーを送信する
  pyautogui.press("enter")

def get_text():
  img = ImageGrab.grab(region)
  img = np.asarray(img)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  image = './Temp/screenshot.png'
  cv2.imwrite(image,img)

  #gray
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  #threshold
  img = cv2.threshold(img,140,255,cv2.THRESH_BINARY)[1]

  #bitwise
  img = cv2.bitwise_not(img)
  image_bitwise = './Temp/target.png'
  cv2.imwrite(image_bitwise,img)

  # OCRエンジンを取得
  engines = pyocr.get_available_tools()
  engine = engines[0]

  # 画像の文字を読み込む
  text = engine.image_to_string(Image.open(image_bitwise), lang="jpn")
  text = text.split('\n\n')[0].replace(" ", "")

  f = open("./Temp/orc.txt", 'w', encoding='UTF-8', newline='\n')

  f.write(text.split('\n\n')[0])
  f.close()