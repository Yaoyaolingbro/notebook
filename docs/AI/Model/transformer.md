# Transformer
<!-- prettier-ignore-start -->
!!! note "摘要"

    论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762v7)
    
    参考资料：http://jalammar.github.io/illustrated-transformer/
    
    源码参考：https://github.com/jadore801120/attention-is-all-you-need-pytorch 
<!-- prettier-ignore-end -->

## 论文内容批注
1. [BELU](https://en.wikipedia.org/wiki/BLEU)一种自动评估机器翻译文本的指标。BLEU 得分是一个 0 到 1 之间的数字，用于衡量机器翻译文本与一组高质量参考翻译的相似度。0 表示机器翻译的输出与参考翻译没有重叠（低质量），而 1 表示其与参考翻译完全重叠（高质量）。![](graph/BELU.png)

> 事实表明，BLEU 得分与人类对翻译质量的判断有很好的相关性。请注意，即使是人工翻译也无法达到 1.0 的满分。一般来说，>40%就算是很高质量的翻译，>60%则往往超过人工翻译。

2. [LSTM](cnn.md)

## 代码解释
