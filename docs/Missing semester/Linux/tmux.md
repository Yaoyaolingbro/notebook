# tmux使用以及配置

## tmux - plugin
> 官方网址: [tmux-plugins](https://github.com/tmux-plugins/tpm)

### 安装
```bash
git clone git@github.com:tmux-plugins/tpm.git ~/.tmux/plugins/tpm
vim ~/.tmux.conf
```

并将下面的内容添加到`~/.tmux.conf`中
```bash
# List of plugins
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-sensible'

# Other examples:
# set -g @plugin 'github_username/plugin_name'
# set -g @plugin 'github_username/plugin_name#branch'
# set -g @plugin 'git@github.com:user/plugin'
# set -g @plugin 'git@bitbucket.com:user/plugin'
set -g @plugin 'catppuccin/tmux'
set -g @catppuccin_flavour 'mocha'


# Initialize TMUX plugin manager (keep this line at the very bottom of tmux.conf)
run '~/.tmux/plugins/tpm/tpm'

set -g default-terminal 'tmux-256color'
```

最后使用快捷键`ctrlb(prefix) + shift I`安装插件
