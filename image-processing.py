from openai import OpenAI
import os

model = "gpt-4o"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "You are a mathematics teacher who helps me with my math homework."},
        {
            "role": "user",
            "content":
                [
                    {
                        "type": "text",
                        "text": "The square has a side length of 1 unit. Find the area of the circle."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://ibb.co/wRhw01C"}
                    }
                ]
        }
    ],
    temperature=0.0,
)

print(response.choices[0].message.content)