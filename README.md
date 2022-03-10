# LSTM-Numpy
## Description
The implementation of the LSTM network from scratch using Numpy in Python, and the training of a simple word level language model.

This code is an implementation of the LSTM variant called CIFG (Coupled Input and Forget Gates) from the paper `LSTM : A Search Space Odyssey` [link](https://arxiv.org/pdf/1503.04069.pdf)

The model is a two stacked layers of LSTMs followed by a linear classification layer for decoding the next character. It is trained on a shakespeare corpus.

The code also contains a simple implementation of the SGD with momentum algorithm for the network parameters optimization.