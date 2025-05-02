import sys
import threading
from typing import Union
import os, tempfile, sounddevice as sd, numpy as np, webrtcvad
from scipy.io.wavfile import write
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import subprocess

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

with open("SystemPrompt", 'r') as f:
    system_prompt = f.read().replace("\n", "")


def record_until_silence(
        fs: int = 16_000,
        frame_ms: int = 30,
        pad_s: float = 1
) -> str:
    vad = webrtcvad.Vad(2)
    frame_len = int(fs * frame_ms / 1000)
    silence_limit = int(pad_s * 1000 / frame_ms)

    print("🎙  Speak… (don't talk to stop recording)")
    frames = []
    silence_run = 0
    started = False
    stop_event = threading.Event()

    def cb(indata, *_):
        nonlocal silence_run, started
        pcm = indata.tobytes()
        speech = vad.is_speech(pcm, fs)
        if speech:
            started = True
            silence_run = 0
        else:
            if started:
                silence_run += 1
        frames.append(pcm)
        if started and silence_run > silence_limit:
            print("🛑 Silence detected. Stopping...")
            stop_event.set()

    with sd.InputStream(
            channels=1, samplerate=fs, dtype="int16",
            blocksize=frame_len, callback=cb
    ):
        stop_event.wait()

    pcm_bytes = b"".join(frames)
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    write(wav_path, fs, np.frombuffer(pcm_bytes, dtype=np.int16))

    return wav_path


class Model:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        self.context = [SystemMessage(content=system_prompt)]
        self.builder = StateGraph(MessagesState)
        self.build_graph()

    def call_llm(self, state: MessagesState):
        history = state["messages"]
        ai_message = self.llm.invoke(history)
        return {"messages": history + [ai_message]}

    def _trim_history(self):
        system, rest = self.context[0], self.context[1:]
        if len(rest) > 15:
            rest = rest[-15:]
        self.context = [system] + rest

    def add_message_to_context(self, msg: Union[HumanMessage, AIMessage]):
        self.context.append(msg)
        self._trim_history()

    def build_graph(self):
        self.builder.add_node("call_llm", self.call_llm)
        self.builder.add_edge(START, "call_llm")
        self.builder.add_edge("call_llm", END)
        self.graph = self.builder.compile()

    def get_response(self, user_input):
        self.add_message_to_context(HumanMessage(content=user_input))
        state = self.graph.invoke({"messages": self.context})
        self.context = state["messages"]
        return state["messages"][-1].content

    def listen(self) -> str:
        wav = record_until_silence()
        with open(wav, "rb") as f:
            tr = client.audio.transcriptions.create(
                model="whisper-1",
                file=f)
        Path(wav).unlink(missing_ok=True)
        return tr.text.strip()

    @staticmethod
    def speak(text: str, voice: str = "onyx"):
        mp3_path = os.path.join("audios", "temp.mp3")
        face_path = os.path.join("..", "kizaru.webp")

        with client.audio.speech.with_streaming_response.create(
                model="tts-1", voice=voice, input=text
        ) as resp:
            resp.stream_to_file(mp3_path)

        mp3_path = os.path.join("..", "audios", "temp.mp3")

        cmd = [
            sys.executable,
            "inference.py",
            "--enhancer", "gfpgan",
            "--source_image", str(face_path),
            "--driven_audio", str(mp3_path),
        ]

        subprocess.run(cmd, cwd="SadTalker1", env=os.environ.copy())

        os.startfile(os.path.abspath("videos/output.mp4"))
