import gradio as gr
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from elevenlabs import generate, play, set_api_key, clone
import os
import speechmatics
import time
import wave


SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY", "")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "")
set_api_key(ELEVEN_API_KEY)


speechmatics_url = "wss://eu.rt.speechmatics.com/v2/en"
current_transcript = dict()
transcript, prev_transcript, msg = "", "", ""
prev_msg_len = 0
responding = False
timeout = 0
starting = True
voice = "Adam"
operating_point = "enhanced"
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=memory
)

def create_speechmatics_client(speechmatics_url: str):
    conn = speechmatics.models.ConnectionSettings(
        url=speechmatics_url,
        auth_token=SPEECHMATICS_API_KEY,
    )
    return speechmatics.client.WebsocketClient(conn)


def init(sm_client):
    def update_transcript(msg):
        global transcript
        global start
        current_transcript = msg["metadata"]["transcript"].strip()
        if len(current_transcript) > 0:
            transcript += current_transcript + " "

    sm_client.add_event_handler(speechmatics.models.ServerMessageType.AddTranscript, event_handler=update_transcript)

    return True

class RawInputStreamWrapper:
    def __init__(self, audio_file):
        self.wave_object = wave.open(audio_file)

    def read(self, frames):
        return self.wave_object.readframes(frames)

async def transcribe(audio_file):
    global operating_point
    speechmatics_client = create_speechmatics_client(speechmatics_url)
    init(speechmatics_client)
    frame_rate=48000
    settings = speechmatics.models.AudioSettings(
        sample_rate=frame_rate,
        chunk_size=1024,
        encoding="pcm_s16le"
    )
    # Define transcription parameters
    conf = speechmatics.models.TranscriptionConfig(language='en',operating_point=operating_point, max_delay=2, max_delay_mode="flexible", enable_partials=False, enable_entities=False)
    await speechmatics_client.run(RawInputStreamWrapper(audio_file), conf, settings)

def update_language_model(llm_name, prompt):
    global conversation
    global memory
    if prompt != "": 
        memory.chat_memory.add_ai_message(prompt)
    if llm_name[0] == "Claude":
        llm = ChatAnthropic()
    else:
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    conversation = ConversationChain(
        llm=llm, 
        verbose=True, 
        memory=memory
    )

def update_operating_point(op_point):
    global operating_point 
    operating_point = op_point

def check_for_done():
    global transcript
    global prev_transcript
    global responding
    global msg
    global prev_msg_len
    global timeout
    global start
    pause = 0.1
    if transcript == prev_transcript and len(transcript) > 0:
        if timeout == 0:
            timeout = time.time()
        elif time.time() - timeout > pause:
            msg = transcript[prev_msg_len:]
            if len(msg) > 0:
                prev_msg_len += len(msg)
                responding = True
    else:
        timeout = 0
    prev_transcript = transcript

def respond(user_input, bot_response):
    global msg
    global responding
    global start
    global conversation
    if responding:
        responding = False
        user_input = msg
        bot_response = conversation.predict(input=user_input)
    return user_input, bot_response

def output(chat_history, user_input, bot_response):
    global voice
    chat_history.append((user_input, bot_response))
    audio = generate(text=bot_response, voice=voice)
    file = 'audio.wav'
    with open(file, mode='bw') as f: 
        f.write(audio)
    return chat_history, user_input, bot_response, file


def update_voice(file, name, description, progress=gr.Progress()):
    global voice
    name = "NoName" if name == "" else name
    if file != "":
        voice = clone(
                name=name,
                description=description,
                files=[file.name],
            )
    return None,"",""

start = time.time()

with gr.Blocks() as demo:
    gr.Markdown("# Speechmatics Chatbot")
    gr.Markdown("Hello, I'm the Speechmatics chatbot, a simple prototype of an AI that you can have a verbal conversation with. Please be patient with me, I'm only a prototype so I can be a little slow to respond. See [speechmatics.com](https://www.speechmatics.com/) for more information about my underlying ASR technology.")
    gr.Markdown("### Settings")
    with gr.Accordion("Transcription", open=False):
        op_point = gr.Dropdown(choices=["standard", "enhanced"], value=["enhanced"], label="Choose the operating point you'd like to use. Do this before activating your Microphone.", interactive=True)
        op_button = gr.Button(value="Select")
        op_button.click(update_operating_point, [op_point])
    with gr.Accordion("Large Language Model", open=False):
        llm = gr.Dropdown(choices=["GPT-3.5", "Claude"], value=["GPT-4"], label="Select which LLM you'd like to use", interactive=True)
        prompt = gr.Textbox("", label="Conversation Prompt", visible=True)
        llm_button = gr.Button(value="Select")
        llm_button.click(update_language_model, [llm, prompt])
    with gr.Accordion("Speech Synthesis", open=False):
        gr.Markdown("(Optional) Upload an audio file (max. 10MB) to clone the voice of the person you want to speak to. Do this before activating your Microphone.")
        with gr.Group():
            with gr.Row().style(equal_height=True):
                    with gr.Column():
                        voice_name = gr.Textbox("", label="voice name")
                        voice_description = gr.Textbox("", label="voice description")
                    voice_file = gr.File(label="voice file")
        clone_button = gr.Button(value="Clone")
        clone_button.click(update_voice, [voice_file, voice_name, voice_description], [voice_file, voice_name, voice_description])
    gr.Markdown("### Record from your microphone to start chatting!")
    audio = gr.Audio(source="microphone", type="filepath", label="Audio")
    bot_audio = gr.Audio(type="filepath", label="Output Audio", visible=True, autoplay=True)    # currently audio will only autoplay once - bug with gradio. Issue raised: https://github.com/gradio-app/gradio/issues/4549 
    gr.Markdown("Your current chat:")
    chatbot = gr.Chatbot()
    user_input = gr.Textbox("", label="user_input", visible=False)
    bot_response = gr.Textbox("", label="bot_response", visible=False)
    demo.queue()
    audio.stream(transcribe, inputs=[audio])
    audio.stream(check_for_done)
    audio.stream(respond, [user_input, bot_response], [user_input, bot_response])
    user_input.change(output, [chatbot, user_input, bot_response], [chatbot, user_input, bot_response, bot_audio])
    
demo.launch()