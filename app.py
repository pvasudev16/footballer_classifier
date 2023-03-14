#!/usr/bin/env python
# coding: utf-8

import fastbook
import gradio
import PIL
import gradio as gr

learner = fastbook.load_learner("./zidane_scholes_mbappe.pkl")

labels = learner.dls.vocab
labels

def predict(img):
    # img = fastbook.PILImage.create(img)
    pred, pred_idx, probs = learner.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(shape=(512, 512)),
    outputs=gr.outputs.Label(num_top_classes=3)
).launch(
    share=True
)


