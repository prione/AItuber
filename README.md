# AItuber
AItuber with ChatGPT

AIで生成したイラストをAIで動かし、AIで設定したキャラクターがAIによって喋ったりゲームしたりする。全てがAIのAItuberを目指しています。

ChatBotにはChatGPT, 合成音声にはVOICEVOX, アバターにはEasyVtuber( https://github.com/yuyuyzl/EasyVtuber )を使用しています。

*EasyVtuberは一枚のイラストを動かすtalking-head-anime-3をOBSの配信向けに拡張したものです。

<img src=https://github.com/prione/AItuber/assets/92021420/e10b0d1a-83d7-4e8c-aa6f-3446f6bcccc width="50%" />

# DEMO
[![alt設定](http://img.youtube.com/vi/TCTUUorWVzk/0.jpg)](https://www.youtube.com/watch?v=TCTUUorWVzk)

# Installation
### 必要ライブラリ
```
pip install -r requirements.txt
```

### VOICEVOX
1. voicevox_engineをクローン
```
git clone https://github.com/VOICEVOX/voicevox_engine
```
2. クローンしたフォルダををVOICEVOXという名前に変更し、run.batと同じディレクトリに配置

3. https://github.com/VOICEVOX/voicevox_engine/releases/tag/0.14.6 からエンジン本体>Windows（GPU/CUDA版）をダウンロード

4. 解凍して出てきたフォルダをcoreという名前に変更し、VOICEVOXフォルダ内に配置
```
+ Avator
+ Language
+ VOICEVOX
  ...
  - run.py
  + core
    ...
    - voicevox_core.dll
```

### EasyVtuber
https://github.com/pkhungurn/talking-head-anime-3-demo#download-the-models からtalking-head-anime-3モデルをダウンロードし、以下のように配置
```
+ Avator/data/models
  - separable_float
  - separable_half
  - standard_float
  - standard_half
  - placeholder.txt
```

# Usage
run.batを起動
1. (DEBUG): テストモード。アバターの出力先がGUIウィンドウになります。コマンドラインを通じてChatBotに入力をします。

2. (OBS): 配信モード。アバターの出力先が仮想カメラになります。また、Youtubeライブ配信をChatBotの入力として取り込みます。

3. (Only Language): テストモード。ChatBotモジュールだけを起動します。

4. (Only Avator):  テストモード。アバターモジュールだけを起動します。
