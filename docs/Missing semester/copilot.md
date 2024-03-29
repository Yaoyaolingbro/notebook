# Github Copilot
<!-- prettier-ignore-start -->
!!! info "摘要"
    个人认为AI算是21世纪的一个福祉了，客观来说我们有很多没有意义的文章或者文字需要编写。于此同时，AI也能成为一个非常耐心的老师指导你编程。当然学生认证的话，你可以免费使用AI编程工具Github Copilot。这里就简单介绍一下如何注册。

    **Attention：** AI不是万能的，它只是一个工具，它的输出需要你自己去判断是否正确。你的重要代码也请不要通过copilot编写，因为它可能会泄露你的代码！

    - [Github Copilot](https://copilot.github.com/)
    - [Github Copilot注册知乎教程](https://zhuanlan.zhihu.com/p/618772237)
<!-- prettier-ignore-end -->

## 1. 注册
> 1. 个人认为无论阅读任何一个博主的教程,你都需要结合自己的实际情况来做的
> 2. 我是结合上方知乎的教程认证的,故底下只是一个简短的指引.

1. 首先你需要一个GitHub账号，如果你没有的话可以去注册一个。(刚注册完的用户好像得等几天才能申请)
2. 然后你需要一个学生认证，你可以通过学校邮箱认证，也可以通过学生证认证。如果你是国内的话，建议你使用学生证认证，因为国内的学校邮箱很多都是不支持的。
3. 我前两次都没有通过, 最后将学信网上的报告翻译成英文(注意可以添加remote study),即可通过
4. 大概等个三四天的时间你会收到认证成功的邮箱消息,然后打开[github plans and usage](https://github.com/settings/billing/summary)去enable 你的copilot就好啦!

## 2. 使用技巧
1. 你可以自行编写注释，它会根据你的注释生成代码。
2. 你可以编写markdown文件时调用copilot，他真的相当智能（比如我编写离散资料的时候它会50%以上的内容都可以推断正确，大幅度加快你的效率）。当然一些无意义的实验报告也可以通过这种方式编写。
3. 此外你可以下载GitHub copilot chat插件，可以当作一个gpt来使用。同时一个小技巧就是你可以添加 `@workspace`, 它会导入你整个项目作为参考资料
4. 于此同时，如果你使用vscode，相信你一定是会使用terminal的。当你的terminal出现报错，你可以点击左边的星星，他会帮你分析问题。
5. 勾选代码并按下`ctrl+I`，输入`/doc`它会帮你自动添加注释。