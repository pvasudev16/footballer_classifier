---
title: Zidane Scholes Mbappe
emoji: 📉
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 3.20.1
app_file: app.py
pinned: false
license: apache-2.0
---

# Zidane-Scholes-Mbappe Classifier

A simple image recognition model to classify images as being either Kylian Mbappe, Zinedine Zidane, or Paul Scholes.

This model is trained using fastai. Using fastai, I got images of each of the players from DuckDuckGo, augmented the data with rotations, skews, crops, and resizes of the images. I then fine-tuned the Resnet34 model to classify images as being one of the three players. I exported the trained model, which I then uploaded to HuggingFace. To see the training process, see kaggle_training_notebook.ipynb

I initially trained this model by fine-tuning Resnet18, but it made obvious mistakes: it classified Paul Scholes as Mbappe. Moving to Resnet34 makes the test examples work properly
