UNCW Computer-Vision (452) Final Project: Lip Reading CNN

This project is a lip-reading CNN that recognizes spoken words using only video of the mouth, no audio. It uses a combination of convolutional neural networks (CNNs) for visual feature extraction and a long short-term memory (LSTM) layer to model how the mouth moves over time.

We trained the model on custom-recorded datasets of three words: "hello", "goodbye", and "hamburger", each consisting of short video clips of cropped mouth movements. The system processes each 1-second clip (30 frames), extracts frame-by-frame features using a TimeDistributed CNN, and then predicts the spoken word using temporal modeling with an LSTM.

WATCH OUR SHORT DEMO ON TIKTOK: ( https://www.tiktok.com/t/ZTjAhmgNs/ )
