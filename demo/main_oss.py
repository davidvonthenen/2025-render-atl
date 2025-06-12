import time
import os
import threading
import queue
import warnings
import uuid
import requests
import json

from common.types import Message, TextPart, TaskSendParams

import sounddevice as sd

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from kokoro_onnx import Kokoro

from deepgram import (
    Microphone,
    Speaker,
)

import openai


#################
# Speech-to-Text
#################

# Microphone instance.
global mic
mic = None

global speaker
speaker = None

# what device are we using?
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Optionally, to suppress deprecation warnings, uncomment:
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# Thread-safe queue for audio chunks.
audio_queue = queue.Queue()
stop_event = threading.Event()

# Energy threshold: chunks with RMS energy below this are considered silence.
energy_threshold = 0.010

# Silence duration (in seconds) required to mark the end of a thought.
silence_duration = 1.5

# Determine device index for the HF pipeline (-1 for CPU, >=0 for GPU).
pipeline_device = 0 if torch.cuda.is_available() else -1

# Local path to the Whisper model you downloaded.
model_path = "/Users/vonthd/models/whisper-large-v3-turbo"

# Load the Whisper model and processor **from local disk**.
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    local_files_only=True,
)
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

# Initialize the Whisper ASR pipeline using the loaded model, tokenizer, and feature extractor.
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=pipeline_device,
)

def push_audio(data: bytes):
    """
    Callback function for the Microphone.
    Receives raw PCM bytes and enqueues them for processing.
    """
    # print("DEBUG: Received audio chunk of size", len(data))
    audio_queue.put(data)

def process_audio_buffer(silence_duration: float = 1.5):
    """
    Accumulates audio chunks that pass the energy threshold into an utterance buffer.
    When a period of silence (no above-threshold audio) of at least `silence_duration`
    seconds is detected, the utterance is processed via the ASR pipeline.
    """
    utterance_buffer = bytearray()
    sample_rate = 16000  # Must match the microphone's sample rate.
    last_speech_time = None

    while not stop_event.is_set():
        try:
            data = audio_queue.get(timeout=0.1)
        except queue.Empty:
            # If no new data arrives and we have buffered speech, check for silence timeout.
            if utterance_buffer and last_speech_time is not None:
                if time.time() - last_speech_time >= silence_duration:
                    # print("DEBUG: Silence detected due to timeout. Processing utterance buffer...")
                    _process_buffer(utterance_buffer, sample_rate)
                    utterance_buffer = bytearray()
                    last_speech_time = None
            continue

        # Convert the incoming chunk to a numpy array and compute its RMS energy.
        chunk_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        chunk_energy = np.sqrt(np.mean(chunk_np ** 2))
        # print(f"DEBUG: Chunk RMS energy: {chunk_energy:.5f}")

        if chunk_energy >= energy_threshold:
            # Speech detected: add the chunk to the buffer and update the timestamp.
            utterance_buffer.extend(data)
            last_speech_time = time.time()
        else:
            # Chunk is silent; if we already have buffered speech, check if it's time to process.
            if utterance_buffer and last_speech_time is not None:
                if time.time() - last_speech_time >= silence_duration:
                    # print("DEBUG: Silence detected. Processing utterance buffer...")
                    _process_buffer(utterance_buffer, sample_rate)
                    utterance_buffer = bytearray()
                    last_speech_time = None

    # After stop_event is set, process any remaining buffered audio.
    if utterance_buffer:
        # print("DEBUG: Processing final utterance buffer after stop event...")
        _process_buffer(utterance_buffer, sample_rate)

def _process_buffer(buffer: bytearray, sample_rate: int):
    """
    Helper function to process a complete utterance buffer.
    Converts the raw PCM bytes to a normalized numpy array, computes RMS energy,
    and if above the threshold, transcribes the audio using Whisper.
    """
    audio_np = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
    energy = np.sqrt(np.mean(audio_np ** 2))
    # print(f"DEBUG: Processed buffer RMS energy = {energy:.5f}")

    if energy > energy_threshold:
        result = asr_pipeline(audio_np)
        transcription = result.get("text", "").strip()
        if transcription:
            # print("Final Transcription:", transcription, "\n")
            _process_transcription(transcription)
        # else:
        #     print("Final Transcription: [No speech detected]\n")
    # else:
    #     print("DEBUG: Processed buffer energy below threshold. Likely silence. No transcription emitted.\n")

#################
# create the OpenAI client
#################
openai_client = openai.OpenAI()

# initial messages
openai_messages = [
    {
        "role": "assistant",
        "content": "You are David's personal AI Agent. Limit all responses to 2 concise sentences and no more than 100 words."
        "If this is an acknowledgment to purchasing something, give a positive confirmation.",
    },
]

#################
# Text-to-Speech
#################
# kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
kokoro = Kokoro("kokoro-v1.0.fp16.onnx", "voices-v1.0.bin")

def _speak(text: str):
    """
    Function to convert text to speech using Kokoro and play the audio.
    """
    if not text:
        print("No text to speak.")
        return

    # Generate audio samples from the text.
    samples, sample_rate = kokoro.create(text, voice="af_sarah", speed=1.0, lang="en-us")
    
    # Play the generated audio.
    print("Playing audio...")
    mic.mute()

    # old way using sounddevice
    # sd.play(samples, sample_rate)
    # sd.wait()

    # Convert float32 [-1,1] to int16 PCM bytes expected by Speaker
    pcm_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    # Queue the bytes for playback
    speaker.add_audio_to_queue(pcm_int16)

    # Wait for the audio to finish playing
    speaker.wait_for_complete()

    mic.unmute()


def _process_transcription(transcription: str):
    if str(transcription) == "":
        # print("Empty string. Exit.")
        return
    if len(str(transcription)) < 10:
        # print("String is too short to be an answer. Exit.")
        return

    # keyword trigger
    if "computer" not in transcription.lower():
        print("No Computer")
        return

    # append messages
    print(f"Speech-to-Text: {transcription}")

    # append the user input to the openai messages
    openai_messages.append(
        {"role": "user", "content": transcription}
    )

    # LLM
    completion = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=openai_messages,
    )

    # result
    save_response = completion.choices[0].message.content

    # print the response
    print(f"\nLLM Response: {save_response}\n")

    # append the response to the openai messages
    openai_messages.append(
        {"role": "assistant", "content": f"{save_response}"}
    )

    # play audio
    _speak(save_response)

    # small delay
    time.sleep(0.2)


#################
# Handle recommendations from our agent
#################
COORD_URL = "http://localhost:5050/tasks"
JSONRPC_ID = 1

def build_payload(context: str, question: str):
    msg = Message(
        role="user",
        parts=[TextPart(text=context), TextPart(text=question)]
    )
    params = TaskSendParams(id=uuid.uuid4().hex, message=msg)
    return {
        "jsonrpc": "2.0",
        "id": JSONRPC_ID,
        "method": "tasks/send",
        "params": params.model_dump()
    }

def send(context: str, question: str):
    payload = build_payload(context, question)
    resp = requests.post(COORD_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if data.get("error"):
        print("Coordinator error:", json.dumps(data["error"], indent=2))
        return

    parts = (data["result"]
             ["status"]["message"]["parts"])

    # Flexible printing: one-part response
    # print("\nANSWER:\n", parts[0]["text"])
    return parts[0]["text"]


def _process_found_entity():
    # we are going to call this function on a thread and every 2 seconds we are going to look at the first line of the file
    # and take the entity and call the send function

    while not stop_event.is_set():
        time.sleep(2)

        # read the first line of the file
        try:
            response = ""
            with open("products.txt", "r") as f:
                product = f.readline().strip()
                if product:
                    print(f"Found entity: {product}")

                    ctx = (
                        f"""{product}"""
                    )
                    q = (
                        "If this is an musician, offer a suggestion for concert tickets to purchase. If this is anything else, suggest a product to buy."
                    )
                    response = send(ctx, q)
            
            # truncate the file after reading
            with open("products.txt", "w") as f:
                f.truncate(0)

            # speak the response
            if response and len(response) > 0:
                print(f"Recommendation: {response}")
                _speak(response)

                # append to the openai messages
                openai_messages.append(
                    {"role": "assistant", "content": f"{response}"}
                )

        except FileNotFoundError:
            print("products.txt not found. Skipping.")
        except Exception as e:
            print(f"Error reading file: {e}")


def main():
    # Start the audio processing thread.
    processing_thread = threading.Thread(target=process_audio_buffer, args=(silence_duration,), daemon=True)
    processing_thread.start()

    # Create a Microphone instance with our push_audio callback.
    global mic
    mic = Microphone(push_callback=push_audio)
    if not mic.start():
        print("Failed to start the microphone.")
        return
    
    global speaker
    speaker = Speaker(rate=24000)
    if not speaker.start():
        print("Failed to start the speaker.")
        return

    # Start the entity processing thread.
    entity_thread = threading.Thread(target=_process_found_entity, daemon=True)
    entity_thread.start()
    
    print("Recording... Press Enter to stop.")
    input()  # Wait for user input to stop recording.

    # Clean up: stop the microphone and processing thread.
    mic.finish()
    stop_event.set()

    # Wait for the processing threads to finish.
    entity_thread.join()
    processing_thread.join()

    print("Finished.")


if __name__ == "__main__":
    main()
