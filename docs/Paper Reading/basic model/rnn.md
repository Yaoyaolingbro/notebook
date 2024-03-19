# RNN
循环神经网络的大名相信大家肯定早就已经听过了。我们不妨来看篇[博客](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)怎么说。

## LSTM
An [LSTM](https://www.analyticsvidhya.com/blog/2021/03/introduction-to-long-short-term-memory-lstm/) is a type of recurrent neural network that addresses the vanishing gradient problem in vanilla RNNs through additional cells, input and output gates. Intuitively, vanishing gradients are solved through additional additive components, and forget gate activations, that allow the gradients to flow through the network without vanishing as quickly.

<!-- prettier-ignore-start -->
??? note "资源汇总"
    1. [Guide Blog](https://medium.datadriveninvestor.com/how-do-lstm-networks-solve-the-problem-of-vanishing-gradients-a6784971a577)
    2. [cs224d slide](https://cs224d.stanford.edu/lectures/CS224d-Lecture10.pdf)
    3. [cs224d LSTM & GLU](https://wugh.github.io/posts/2016/03/cs224d-notes4-recurrent-neural-networks-continue/)
<!-- prettier-ignore-end -->