import time
import os
import threading
import queue
import warnings
import uuid
import requests
import json

from common.types import Message, TextPart, TaskSendParams

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
    SpeakWSOptions,
)

import openai


#################
# Speech-to-Text
#################

# Optionally, to suppress deprecation warnings, uncomment:
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# Thread-safe queue for audio chunks.
stop_event = threading.Event()

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
# Speech-to-Text and Text-to-Speech
#################
config = DeepgramClientOptions(
    options={"keepalive": "true", "speaker_playback": "true"},
    # verbose=verboselogs.DEBUG,
)
deepgram: DeepgramClient = DeepgramClient("", config)

# create the Microphone
microphone = Microphone()

# listen
dg_listen_connection = deepgram.listen.websocket.v("1")
dg_speak_connection = deepgram.speak.websocket.v("1")

# on_message
def on_message(self, result, **kwargs):
    # Speech-to-Text
    sentence = result.channel.alternatives[0].transcript
    if len(sentence) < 8:
        return
    if result.is_final is False:
        return

    # keyword trigger
    if "computer" not in sentence.lower():
        print("No Computer")
        return

    # append messages
    # print(f"Speech-to-Text: {sentence}")

    # append the user input to the openai messages
    openai_messages.append(
        {"role": "user", "content": sentence}
    )

    # mute the microphone
    microphone.mute()

    save_response = ""
    try:
        for response in openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=openai_messages,
            stream=True,
        ):
            # here is the streaming response
            for chunk in response:
                if chunk[0] == "choices":
                    llm_output = chunk[1][0].delta.content

                    # skip any empty responses
                    if llm_output is None or llm_output == "":
                        continue

                    # save response and append to buffer
                    save_response += llm_output

                    # send to Deepgram TTS
                    dg_speak_connection.send_text(llm_output)

        print(f"Text-to-Speech: {save_response}\n\n")
        openai_messages.append(
            {"role": "assistant", "content": f"{save_response}"}
        )
        dg_speak_connection.flush()

    except Exception as e:
        print(f"LLM Exception: {e}")

    # wait for audio completion
    dg_speak_connection.wait_for_complete()

    # unmute the microphone
    microphone.unmute()

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
                # mute the microphone
                microphone.mute()

                # send to Deepgram TTS
                dg_speak_connection.send_text(response)

                print(f"Assistant Recommendation: {response}\n\n")

                # append to the openai messages
                openai_messages.append(
                    {"role": "assistant", "content": f"{response}"}
                )
                dg_speak_connection.flush()

                # wait for audio completion
                dg_speak_connection.wait_for_complete()

                # unmute the microphone
                microphone.unmute()

        except FileNotFoundError:
            print("products.txt not found. Skipping.")
        except Exception as e:
            print(f"Error reading file: {e}")


def main():
    dg_listen_connection.on(LiveTranscriptionEvents.Transcript, on_message)

    # start speak connection
    speak_options: SpeakWSOptions = SpeakWSOptions(
        model="aura-luna-en",
        encoding="linear16",
        sample_rate=16000,
    )
    dg_speak_connection.start(speak_options)

    # start listen connection
    listen_options: LiveOptions = LiveOptions(
        model="nova-2-conversationalai",
        punctuate=True,
        smart_format=True,
        language="en-US",
        encoding="linear16",
        numerals=True,
        channels=1,
        sample_rate=16000,
        interim_results=True,
        utterance_end_ms="2000",
    )
    dg_listen_connection.start(listen_options)

    # set the callback on the microphone object
    microphone.set_callback(dg_listen_connection.send)

    # start microphone
    microphone.start()

    # Start the entity processing thread.
    entity_thread = threading.Thread(target=_process_found_entity, daemon=True)
    entity_thread.start()
    
    print("Recording... Press Enter to stop.")
    input()  # Wait for user input to stop recording.

    # Wait for the microphone to close
    microphone.finish()

    # Indicate that we've finished
    dg_listen_connection.finish()
    dg_speak_connection.finish()

    print("Finished.")


if __name__ == "__main__":
    main()
