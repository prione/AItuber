import pytchat
import time

class live():
  def __init__(self, commentQ, video_id):
    self.livechat = pytchat.create(video_id = video_id)
    self.queue = commentQ

  def get_comment(self):

    while self.livechat.is_alive():
      chatdata = self.livechat.get()
      for c in chatdata.items:
        self.queue.put(c.message)
      time.sleep(5)

