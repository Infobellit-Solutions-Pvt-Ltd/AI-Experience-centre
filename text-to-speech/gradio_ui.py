import gradio as gr

from utils import generate_audio

demo = gr.Interface(
    title="TextT-To-Speech Application",
    theme = gr.themes.Glass(),
    fn=generate_audio,
    inputs=[gr.components.Textbox(label='Input text'), gr.components.Dropdown(
        label='Voice preset', 
        choices=['v2/en_speaker_0', 'v2/hi_speaker_0','v2/en_speaker_1','v2/ja_speaker_0','v2/es_speaker_0'])],
    outputs=gr.components.Audio(label='Generated audio'),
    allow_flagging='never'
    ).queue(default_concurrency_limit=5)

if __name__ == '__main__':
    demo.launch(share=True)
