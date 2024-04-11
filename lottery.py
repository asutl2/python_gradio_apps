import gradio as gr
import random

def get_fortune(your_name: str):
    fortune_lists = ['大吉','吉','小吉','凶','末吉','大幸福']
    for _ in range(9):
        fortune_lists.append('大吉')
    for _ in range(19):
        fortune_lists.append('吉')
    for _ in range(29):
        fortune_lists.append('小吉')
    for _ in range(19):
        fortune_lists.append('末吉')
    for _ in range(9):
        fortune_lists.append('凶')
    fortune_result = random.choice(fortune_lists)

    return your_name + "さんの今日の運勢は・・・" + fortune_result + "です！"

iface = gr.Interface(fn=get_fortune, inputs="text", outputs="text")
iface.launch()