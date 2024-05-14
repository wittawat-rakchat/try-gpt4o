from openai import OpenAI
import os

model = "gpt-4o"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
completion = client.chat.completions.create(
  model=model,
  messages=[
      {
          "role": "system",
          "content": """ You are a helpful AI Japanese teacher designed to assist non-native Japanese speakers  
          in improving their written Japanese skills. Make sure you explain in English."""
      },
      {
          "role": "user",
          "content": "こんにち！おけんきですか？私は日本を大好きです！"
      }
  ]
)
print("Assistant: " + completion.choices[0].message.content)
