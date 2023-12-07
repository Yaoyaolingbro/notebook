# python环境管理

## pip 
1. 如果需要更新所有pip的包可以使用`pip3 list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip3 install -U`
2. 想要导出所有pip资源到requirement.txt中，你可以使用`pip freeze > requirements.txt`


## conda

## venv