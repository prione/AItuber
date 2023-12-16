from openai import OpenAI
import numpy as np
import copy
import os
import json

class bot():
  
  def __init__(self, character_template_path, do_memorory=True, do_save=True, load_memory=False):

    self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    with open(character_template_path, "r", encoding="utf-8") as f:
      system_settings = f.read()
      self.system_prompt = [{"role": "system", "content": system_settings}]

    self.memory_path = f"{os.path.dirname(__file__)}/memory"
    if load_memory:
      with open(f"{self.memory_path}/log.json", encoding="SHIFT_JIS") as f:
        self.log = json.load(f)
    else:
      self.log =[]

    self.do_memorory = do_memorory
    self.do_save = do_save

    self.tools = [
      {
        "type": "function",
        "function": {
          "name": "get_emotion",
          "description": "気持ちを出力します",
          "parameters": {
            "type": "object",
            "properties": {
              "emotion": {
                "type": "string", 
                "description": "あなたの気持ちです。通常・喜び・悲しみ・恐怖のいずれか一つを選択してください",
              },
            },
          },
        },
      }
    ]



  def reply(self, input):

    prompt = self.create_prompt(input)

    response = self.client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=prompt,
      # functions=self.tools,
      # function_call={"name": "get_emotion"},
      # tools=self.tools,
      # tool_choice={"name": "get_emotion"},
      max_tokens=1024,
      temperature=1,
      top_p=1,
      presence_penalty=1.0,
      frequency_penalty=1.0,
      stream=True
    )

    # response_message = response.choices[0].message
    # tool_calls = response_message.tool_calls

    # if tool_calls:
    #     for tool_call in tool_calls:
    #         function_name = tool_call.function.name
    #         function_args = json.loads(tool_call.function.arguments)
    #         emotion=function_args.get("emotion"),
    #         messages.append(
    #             {
    #                 "tool_call_id": tool_call.id,
    #                 "role": "tool",
    #                 "name": function_name,
    #                 "content": emotion,
    #             }
    #         ) 

    message = ""
    for chunk in response:
      if chunk.choices[0].delta.content is not None:
        content = chunk.choices[0].delta.content
        message += content 
        yield content

      # if chunk.choices[0].delta.function_call is not None:
      #   try:
      #     yield json.loads(a)
      #   except:
      #     try:
      #       json.loads(a)
      #   content = chunk.choices[0].delta.function_call.arguments 

    if self.do_memorory:
      self.memorize(input, message)

      if self.do_save:
        if len(self.log) % 10 == 0:
          self.save()


  def create_prompt(self, input):

    prompt = copy.copy(self.system_prompt)

    if self.log:
      simiar_log = self.similarity_search(input, threshold=0.7, top_k=5)

      for log  in simiar_log:
        prompt.append({"role": "user", "content": log["user"]})
        prompt.append({"role": "assistant", "content": log["assistant"]})
        pass

    prompt.append({"role": "user", "content": input})
    
    return prompt


  def similarity_search(self, input, threshold, top_k):
    embedding = self.client.embeddings.create(input=[input], model="text-embedding-ada-002").data[0].embedding

    similarities = []
    for log in self.log:
      similarity = self.cosine_similarity(embedding, log["score"])
      
      if threshold <= similarity:
        similarities.append([similarity, log])

    similarities.sort(key=lambda i: i[0], reverse=True)

    return [s[1] for s in similarities][:top_k]


  def cosine_similarity(self, a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


  def memorize(self, input, output):

    embedding = self.client.embeddings.create(input=[input+"\n\n"+output], model="text-embedding-ada-002").data[0].embedding
    self.log.append(
      {
        "user": input,
        "assistant": output,
        "score": embedding,
      }
    )


  def save(self):

    if os.path.exists(f"{self.memory_path}/log.json"):
      os.rename(f"{self.memory_path}/log.json", f"{self.memory_path}/bk.json")

    with open(f"{self.memory_path}/log.json", "w") as f:
      json.dump(self.log, f, ensure_ascii=False, separators=(",", ":"))

    if os.path.exists(f"{self.memory_path}/bk.json"):
      os.remove(f"{self.memory_path}/bk.json")    
