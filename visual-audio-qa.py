import os
import cv2
from openai import OpenAI
from moviepy.editor import VideoFileClip
import base64

model = "gpt-4o"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
VIDEO_PATH = "how-bouncing-betty’s-work.mp4"

QUESTION = "Question: How bouncing betty’s work?"


def process_video(video_path, seconds_per_frame=2):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()

    print(f"Extracted {len(base64Frames)} frames")
    print(f"Extracted audio to {audio_path}")
    return base64Frames, audio_path


base64Frames, audio_path = process_video(VIDEO_PATH, seconds_per_frame=1)

# Summarisation: Audio Summary
transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=open(audio_path, "rb"),
)

qa_visual_response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "Use the video to answer the provided question. Respond in Markdown."},
        {
            "role": "user",
            "content": [
                "These are the frames from the video.",
                *map(lambda x: {
                    "type": "image_url",
                    "image_url": {
                        "url": f'data:image/jpg;base64,{x}',
                        "detail": "low"
                    }
                }, base64Frames),
                QUESTION
            ],
        }
    ],
    temperature=0
)

print("Visual QA:\n" + qa_visual_response.choices[0].message.content)

# Q&A: Audio Q&A
qa_audio_response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content":"Use the transcription to answer the provided question. Respond in Markdown."
        },
        {
            "role": "user",
            "content": f"The audio transcription is: {transcription.text}. \n\n {QUESTION}"
        },
    ],
    temperature=0
)

print("Audio QA:\n" + qa_audio_response.choices[0].message.content)

# Q&A: Visual + Audio Q&A
qa_both_response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content":"Use the video and transcription to answer the provided question."
        },
        {
            "role": "user",
            "content": [
                "These are the frames from the video.",
                *map(lambda x: {
                    "type": "image_url",
                    "image_url": {
                        "url": f'data:image/jpg;base64,{x}',
                        "detail": "low"
                    }
                }, base64Frames),
                {
                    "type": "text",
                    "text": f"The audio transcription is: {transcription.text}"},
                QUESTION
            ],
        }
    ],
    temperature=0
)

print("Both QA:\n" + qa_both_response.choices[0].message.content)
