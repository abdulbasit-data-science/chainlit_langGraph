from src.graph import graph
from typing import Dict, Optional
import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig

import os
from dotenv import load_dotenv
import numpy as np
import wave
import io
from groq import Groq
import audioop
load_dotenv()

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY must be set in environment variables")
client = Groq(api_key=GROQ_API_KEY)

SILENCE_THRESHOLD = 3500  # Adjust based on your audio level
SILENCE_TIMEOUT = 1300.0

@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    return default_user

@cl.on_chat_resume
async def on_chat_resume(thread):
    pass



@cl.step(type="tool")
async def speech_to_text(audio_file):
    try:
        filename, file_content, mime_type = audio_file
        audio_buffer = io.BytesIO(file_content)
        audio_buffer.name = filename
        
        response = client.audio.transcriptions.create(
            file=audio_buffer,
            model="whisper-large-v3"
        )
        
        return response.text
    except Exception as e:
        print(f"Error in speech-to-text: {e}")
        return "Failed to transcribe audio."

@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("silent_duration_ms", 0)
    cl.user_session.set("is_speaking", False)
    cl.user_session.set("audio_chunks", [])
    return True
@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    audio_chunks = cl.user_session.get("audio_chunks")

    if audio_chunks is not None:
        audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
        audio_chunks.append(audio_chunk)

    # If this is the first chunk, initialize timers and state
    if chunk.isStart:
        cl.user_session.set("last_elapsed_time", chunk.elapsedTime)
        cl.user_session.set("is_speaking", True)
        return

    audio_chunks = cl.user_session.get("audio_chunks")
    last_elapsed_time = cl.user_session.get("last_elapsed_time")
    silent_duration_ms = cl.user_session.get("silent_duration_ms")
    is_speaking = cl.user_session.get("is_speaking")

    # Calculate the time difference between this chunk and the previous one
    time_diff_ms = chunk.elapsedTime - last_elapsed_time
    cl.user_session.set("last_elapsed_time", chunk.elapsedTime)

    # Compute the RMS (root mean square) energy of the audio chunk
    audio_energy = audioop.rms(
        chunk.data, 2
    )  # Assumes 16-bit audio (2 bytes per sample)

    if audio_energy < SILENCE_THRESHOLD:
        # Audio is considered silent
        silent_duration_ms += time_diff_ms
        cl.user_session.set("silent_duration_ms", silent_duration_ms)
        if silent_duration_ms >= SILENCE_TIMEOUT and is_speaking:
            cl.user_session.set("is_speaking", False)
            await process_audio()
    else:
        # Audio is not silent, reset silence timer and mark as speaking
        cl.user_session.set("silent_duration_ms", 0)
        if not is_speaking:
            cl.user_session.set("is_speaking", True)

async def process_audio():
    # Get the audio buffer from the session
    if audio_chunks := cl.user_session.get("audio_chunks"):
        # Concatenate all chunks
        concatenated = np.concatenate(list(audio_chunks))

        # Create an in-memory binary stream
        wav_buffer = io.BytesIO()

        # Create WAV file with proper parameters
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(24000)  # sample rate (24kHz PCM)
            wav_file.writeframes(concatenated.tobytes())

        # Reset buffer position
        wav_buffer.seek(0)

        cl.user_session.set("audio_chunks", [])

        wav_file = wave.open(wav_buffer, "rb")
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()

        duration = frames / float(rate)
        if duration <= 0.5:
            print("The audio is too short, please try again.")
            return

        audio_buffer = wav_buffer.getvalue()

        input_audio_el = cl.Audio(content=audio_buffer, mime="audio/wav")

        whisper_input = ("audio.wav", audio_buffer, "audio/wav")
        transcription = await speech_to_text(whisper_input)

        # Send the transcribed message to the UI
        message = cl.Message(
            author="You",
            type="user_message",
            content=transcription,
            elements=[input_audio_el],
        )
        await message.send()

        # Process the transcribed message using your existing graph
        await process_message_content(transcription)
async def process_message_content(content):
    # Create a response message
    answer = cl.Message(content="")
    await answer.send()

    # Set up the config for the graph
    config: RunnableConfig = {
        "configurable": {"thread_id": cl.context.session.thread_id}
    }

    # Process the message with your graph
    text_response = ""
    for msg, *_ in graph.stream(
        {"messages": [HumanMessage(content=content)]},
        config,
        stream_mode="messages",
    ):
        if isinstance(msg, AIMessageChunk):
            text_response += msg.content
            answer.content = text_response
            await answer.update()
@cl.on_message
async def main(message: cl.Message):
    # Process text messages normally
    await process_message_content(message.content)