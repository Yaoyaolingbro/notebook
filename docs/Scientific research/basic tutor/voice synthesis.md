语音合成入门
深度学习（20天）
动手学深度学习 https://tangshusen.me/Dive-into-DL-PyTorch/#/ 跟着教程跑一遍代码 （7天）
跟着教程跑，代码自己敲一遍，教程中提到的论文都要看，可以在colab上跑（7天）
教程：https://pytorch.org/tutorials/

需要看的内容：

Deep Learning with PyTorch: A 60 Minute Blitz
Learning PyTorch with Examples
What is torch.nn really?
Visualizing Models, Data, and Training with TensorBoard
Text全部
Audio全部
看懂Transformer （5天）
论文：Attention Is All You Need
参考资料：http://jalammar.github.io/illustrated-transformer/
源码参考：https://github.com/jadore801120/attention-is-all-you-need-pytorch （可以不用跑，但是需要结合论文理解模型内部的模块）
语音合成理论知识（3天）
语音合成 TTS (Text-To-Speech) 的原理是什么？
https://www.zhihu.com/question/26815523/answer/220693948
https://zhuanlan.zhihu.com/p/113282101
Tacotron&Tacotron2——基于深度学习的端到端语音合成模型：https://zhuanlan.zhihu.com/p/101064153
Understanding the Mel Spectrogram：https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53
https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
下载一个adobe audition，拖进去一个语音的音频，感受一下波形和频谱，数据可以从 https://keithito.com/LJ-Speech-Dataset 下载， 这也是我们做实验最常用的一个英文数据集
Understanding Tansformer: http://jalammar.github.io/illustrated-transformer/
什么是共振峰：https://www.zhihu.com/question/24190826
语音常用python库
大概看下tutorial即可，用于下面跑模型时数据处理部分的理解和参考

librosa https://librosa.github.io/librosa/
语音合成模型
任务：阅读论文，把源码运行起来（做完前面任务以后，我会给大家发放GPU）

Tacotron1 （5天）
Paper: Tacotron: Towards End-to-End Speech Synthesis
教程和代码：手把手教程：https://zhuanlan.zhihu.com/p/114212581
Tacotron2 （5天）
Paper: Natural TTS Synthesis by Conditioning Wavenet on MEL Spectrogram Predictions
代码：https://github.com/NVIDIA/tacotron2
TransformerTTS
Paper: Neural Speech Synthesis with Transformer Network
代码： (私有代码，完成上述两个任务以后开放)
FastSpeech
Paper: FastSpeech: Fast, Robust and Controllable Text to Speech
代码：(私有代码，完成上述两个任务以后开放)
FastSpeech 2
Paper: https://arxiv.org/abs/1905.09263
声码器模型
任务：阅读论文，把源码运行起来（不用训练，直接跑inference部分，将梅尔频谱转换成波形）

WaveNet
WaveGlow （5天）
Paper: A Flow-based Generative Network for Speech Synthesis
代码：https://github.com/NVIDIA/waveglow
ParallelWaveGAN （5天）
Paper: Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram
代码：https://github.com/kan-bayashi/ParallelWaveGAN
其他资料
https://erogol.com/text-speech-deep-learning-architectures/

https://github.com/erogol/TTS-papers

Reference Encoder: Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis



*歌声合成
Adversarially Trained Multi-Singer Sequence-To-Sequence Singing Synthesizer: https://arxiv.org/abs/2006.10317

DeepSinger : Singing Voice Synthesis with Data Mined From the Web: https://arxiv.org/abs/2007.04590



