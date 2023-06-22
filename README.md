# Zidane-Scholes-Mbappe Classifier
A simple image recognition Gradio app to classify images as being either
Kylian Mbappe, Zinedine Zidane, or Paul Scholes. See the
[live version](https://huggingface.co/spaces/pvasudev/zidane_scholes_mbappe).

This model is trained using fastai. I got images of each of the players
from DuckDuckGo, augmented the data with rotations, skews, crops, and
resizes of the images. I then fine-tuned the Resnet34 model to classify
images as being one of the three players. I exported the trained model,
which I then uploaded to HuggingFace.
To see the training process, see
[zidane_scholes_mbappe_training.ipynb](https://github.com/pvasudev16/footballer_classifier/blob/main/zidane_scholes_mbappe_training.ipynb).

I initially trained this model by fine-tuning Resnet18, but it made
obvious mistakes: it classified Paul Scholes as Mbappe. 
Moving to Resnet34 makes the test examples work properly. (The model
.pkl files aren't included here.)

I prototyped this app locally using the
[prototype.ipynb](https://github.com/pvasudev16/footballer_classifier/blob/main/prototype.ipynb)
notebook.
