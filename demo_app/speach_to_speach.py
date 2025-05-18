import os
import io
import wave
import numpy as np
import audioop
import chainlit as cl
from typing import Dict, Optional
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from fastrtc import get_stt_model, get_tts_model
from src.graph import graph
from dotenv import load_dotenv

load_dotenv()

# Initialize open-source STT and TTS models
stt_model = get_stt_model()  # Assumes FastRTC provides a Whisper-based model
tts_model = get_tts_model()  # Assumes FastRTC provides a VITS or similar TTS model

# Audio processing settings
SILENCE_THRESHOLD = 3500  # RMS threshold for silence detection
SILENCE_TIMEOUT = 1300.0  # Silence duration (ms) to end a turn
SAMPLE_RATE = 24000  # Audio sample rate (Hz), adjust if TTS model requires different
CHANNELS = 1  # Mono audio
SAMPLE_WIDTH = 2  # 16-bit audio

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



@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("silent",cl.user_session.set("silent_duration_ms", 0))
    cl.user_session.set("is_speaking", False)
    cl.user_session.set("audio_chunks", [])
    return True

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    audio_chunks = cl.user_session.get("audio_chunks")

    # Convert chunk to numpy array
    audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
    audio_chunks.append(audio_chunk)

    if chunk.isStart:
        cl.user_session.set("last_elapsed_time", chunk.elapsedTime)
        cl.user_session.set("is_speaking", True)
        return

    last_elapsed_time = cl.user_session.get("last_elapsed_time")
    silent_duration_ms = cl.user_session.get("silent_duration_ms")
    is_speaking = cl.user_session.get("is_speaking")

    # Calculate time difference
    time_diff_ms = chunk.elapsedTime - last_elapsed_time
    cl.user_session.set("last_elapsed_time", chunk.elapsedTime)

    # Compute RMS energy for silence detection
    audio_energy = audioop.rms(chunk.data, SAMPLE_WIDTH)

    if audio_energy < SILENCE_THRESHOLD:
        silent_duration_ms += time_diff_ms
        cl.user_session.set("silent_duration_ms", silent_duration_ms)
        if silent_duration_ms >= SILENCE_TIMEOUT and is_speaking:
            cl.user_session.set("is_speaking", False)
            await process_audio()
    else:
        cl.user_session.set("silent_duration_ms", 0)
        if not is_speaking:
            cl.user_session.set("is_speaking", True)

async def process_audio():
    audio_chunks = cl.user_session.get("audio_chunks")
    if not audio_chunks:
        return

    # Concatenate audio chunks
    concatenated = np.concatenate(audio_chunks)

    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(concatenated.tobytes())

    wav_buffer.seek(0)
    cl.user_session.set("audio_chunks", [])

    # Check audio duration
    frames = len(concatenated)
    duration = frames / float(SAMPLE_RATE)
    if duration <= 1.0:
        await cl.Message(content="Audio too short, please try again.").send()
        return

    # Read WAV file to extract audio data for STT
    wav_buffer.seek(0)
    with wave.open(wav_buffer, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        audio_frames = wav_file.readframes(wav_file.getnframes())
        # Convert audio frames to NumPy array (16-bit PCM)
        audio_np = np.frombuffer(audio_frames, dtype=np.int16)

    # Prepare input for stt_model.stt
    audio_input = (sample_rate, audio_np)

    try:
        # Transcribe audio using open-source STT
        transcription = stt_model.stt(audio_input)
    except Exception as e:
        await cl.Message(content=f"Error transcribing audio: {str(e)}").send()
        return

    # Display transcribed message
    wav_buffer.seek(0)  # Reset buffer for display
    input_audio_el = cl.Audio(content=wav_buffer.getvalue(), mime="audio/wav")
    await cl.Message(
        author="You",
        type="user_message",
        content=transcription,
        elements=[input_audio_el],
    ).send()

    # Process transcription through the graph
    answer = cl.Message(content="")
    await answer.send()

    config: RunnableConfig = {
        "configurable": {"thread_id": cl.context.session.thread_id}
    }

    response_content = ""
    for msg, _ in graph.stream(
        {"messages": [HumanMessage(content=transcription)]},
        config,
        stream_mode="messages",
    ):
        if isinstance(msg, AIMessageChunk):
            response_content += msg.content
            answer.content = response_content
            await answer.update()

    # Convert response to speech using open-source TTS
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        try:
            # Collect integer samples into a NumPy array
            samples = []
            for chunk in tts_model.stream_tts_sync(response_content):
                if isinstance(chunk, tuple):
                    # Assume tuple is (sample, metadata)
                    sample = chunk[0]
                    # Debugging: Log the first few chunks to inspect format
                    if len(samples) < 5:
                        print(f"TTS chunk: {chunk}, Sample type: {type(sample)}")
                    if isinstance(sample, int):
                        samples.append(sample)
                    elif isinstance(sample, np.ndarray):
                        samples.extend(sample.tolist())
                    elif isinstance(sample, bytes):
                        wav_file.writeframes(sample)
                        continue
                    else:
                        raise ValueError(f"Expected int, NumPy array, or bytes, got {type(sample)}")
                else:
                    # Handle non-tuple chunks (e.g., bytes)
                    if isinstance(chunk, bytes):
                        wav_file.writeframes(chunk)
                    else:
                        raise ValueError(f"Expected bytes for non-tuple chunk, got {type(chunk)}")
            
            # Convert collected samples to NumPy array and write to WAV
            if samples:
                audio_data = np.array(samples, dtype=np.int16).tobytes()
                wav_file.writeframes(audio_data)
        except Exception as e:
            await cl.Message(content=f"Error generating speech: {str(e)}").send()
            return

    audio_buffer.seek(0)
    output_audio_el = cl.Audio(
        auto_play=True,
        mime="audio/wav",
        content=audio_buffer.getvalue(),
    )

    # Update the message with the audio element
    await cl.Message(content=response_content, elements=[output_audio_el]).send()

@cl.on_message
async def main(message: cl.Message):
    # Handle text input
    answer = cl.Message(content="")
    await answer.send()

    config: RunnableConfig = {
        "configurable": {"thread_id": cl.context.session.thread_id}
    }

    response_content = ""
    for msg, _ in graph.stream(
        {"messages": [HumanMessage(content=message.content)]},
        config,
        stream_mode="messages",
    ):
        if isinstance(msg, AIMessageChunk):
            response_content += msg.content
            answer.content = response_content
            await answer.update()

    # Convert response to speech
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        try:
            # Collect integer samples into a NumPy array
            samples = []
            for chunk in tts_model.stream_tts_sync(response_content):
                if isinstance(chunk, tuple):
                    # Assume tuple is (sample, metadata)
                    sample = chunk[0]
                    if isinstance(sample, int):
                        samples.append(sample)
                    elif isinstance(sample, np.ndarray):
                        samples.extend(sample.tolist())
                    elif isinstance(sample, bytes):
                        wav_file.writeframes(sample)
                        continue
                    else:
                        raise ValueError(f"Expected int, NumPy array, or bytes, got {type(sample)}")
                else:
                    # Handle non-tuple chunks (e.g., bytes)
                    if isinstance(chunk, bytes):
                        wav_file.writeframes(chunk)
                    else:
                        raise ValueError(f"Expected bytes for non-tuple chunk, got {type(chunk)}")
            
            # Convert collected samples to NumPy array and write to WAV
            if samples:
                audio_data = np.array(samples, dtype=np.int16).tobytes()
                wav_file.writeframes(audio_data)
        except Exception as e:
            await cl.Message(content=f"Error generating speech: {str(e)}").send()
            return

    audio_buffer.seek(0)
    output_audio_el = cl.Audio(
        auto_play=True,
        mime="audio/wav",
        content=audio_buffer.getvalue(),
    )

    # Update the message with the audio element
    await cl.Message(content=response_content, elements=[output_audio_el]).send()