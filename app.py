from fastai.vision.all import *
import gradio as gr

learner = load_learner("./zidane_scholes_mbappe_resnet34.pkl")

labels = learner.dls.vocab

def predict(img):
    # img = fastbook.PILImage.create(img)
    pred, pred_idx, probs = learner.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(shape=(512, 512)),
    outputs=gr.outputs.Label(num_top_classes=3),
    examples=[
        "zidane_01.jpg",
        "zidane_02.jpg",
        "scholes_01.jpg",
        "scholes_02.jpg",
        "mbappe_01.jpg",
        "mbappe_02.jpg"
    ],
    title="A simple app to differentiate pictures of three amazing footballers",
    description="Classify a picture of a footballer being either Kylian Mbappe, Zinedine Zidane, or Paul Scholes"
).launch()


