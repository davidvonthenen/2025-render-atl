# RenderATL Demo: Agentic AI Voice Assistant

There are two versions of this Agentic AI Voice Assistant:

- Completely Using Open Source Components
- Using the [Deepgram](https://deepgram.com/) Speech-to-Text and Text-to-Speech platform which is more GPU friendly (as in not needing one)

## Python Environment Needed for Both Versions

Python 3.12+ required.

I highly recommend using [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or [venv](https://docs.python.org/3/library/venv.html).

Python

```bash
pip install -r requirements.txt
```

You need to `git clone` the [google Aget2Agent repo](https://github.com/google-a2a/A2A) and do a `pip install -e .`

For MacOS/Linux assuming you have already installed xcode developer tools, this also requires brew installing for the Microphone capabilities:

- portaudio

Set the following environment variables:

- `OPENAI_API_KEY`
- `BESTBUY_API_KEY`
- `TICKETMASTER_API_KEY`

### Best Buy Platform API

Visit [Best Buy Developer Platform](https://developer.bestbuy.com/) and create an account.

### Ticketmaster Platform API

Visit [Ticketmaster Developer Platform](https://developer.ticketmaster.com/) and create an account.

## Option 1: Pure Open Source Version of the Demo

This is a Voice AI Assistant using open source components for STT and TTS.

### Speech-to-Text: OpenAI whisper-large-v3-turbo

You can either dynamically download the `openai/whisper-large-v3-turbo` model as it currently does in the code (default) OR download the model from huggingface here:

- https://huggingface.co/openai/whisper-large-v3-turbo/tree/main

### Text-to-Speech: Kokoro + onnx

Using Kokoro + onnx. Details:

- https://huggingface.co/spaces/hexgrad/Kokoro-TTS
- https://github.com/thewh1teagle/kokoro-onnx

#### For Kokoro + ONNX TTS

Download the following files from their GitHub page:

- kokoro-v1.0.onnx
- voices-v1.0.bin

### Setup and Running the Demo

Set your MacOS Proxy for both HTTP and HTTPS to 127.0.0.1 and port 8080.

In 5 different consoles, run:

- ./run.sh
- python bestbuy_agent.py.py
- python ticketmaster_agent.py.py
- python coordinator_agent.py
- python main_oss.py

## Option 2: Non-GPU Version of the Demo Using Deepgram

This is going to run better on a system without a GPU.

### Deepgram Platform

Visit [Deepgram](https://www.deepgram.com/) and create an account.

### Setup and Running the Demo

Set your MacOS Proxy for both HTTP and HTTPS to 127.0.0.1 and port 8080.

In 5 different consoles, run:

- ./run.sh
- python bestbuy_agent.py.py
- python ticketmaster_agent.py.py
- python coordinator_agent.py
- python main_oss.py
