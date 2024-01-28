
# # 用于将mkdocs.yml文件中的'\'替换为'/'
# import os
# # 读取mkdocs.yml文件
# with open('mkdocs.yml', 'r') as file:
#     file_content = file.read()

# # 替换所有的'\'为'/'
# file_content = file_content.replace('\\', '/')

# # 写入替换后的内容回mkdocs.yml文件
# with open('mkdocs.yml', 'w') as file:
#     file.write(file_content)

# # 用于将docs文件夹下的所有markdown文件中的'\'替换为'/'
# import os

# # 定义替换函数
# def replace_in_file(file_path, old_str, new_str):
#     with open(file_path, 'r') as file:
#         file_content = file.read()
    
#     file_content = file_content.replace(old_str, new_str)
    
#     with open(file_path, 'w') as file:
#         file.write(file_content)

# # 遍历docs文件夹下的所有markdown文件
# docs_folder = 'docs'
# old_str = '\\Snipaste'
# new_str = '/Snipaste'

# for foldername, subfolders, filenames in os.walk(docs_folder):
#     for filename in filenames:
#         if filename.endswith('.md'):
#             file_path = os.path.join(foldername, filename)
#             replace_in_file(file_path, old_str, new_str)

# 替换图片格式
import os
import re

# 定义docs文件夹路径
folder_path = 'docs'

# 遍历文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".md"):
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                content = f.read()
                # 使用正则表达式匹配图片标签
                img_tags = re.findall(r'<img alt="([^"]+)" src="([^"]+)">', content)
                for alt, src in img_tags:
                    # 生成Markdown图片语法
                    markdown_img = f'![{alt}]({src})'
                    # 替换原始图片标签
                    content = content.replace(f'<img alt="{alt}" src="{src}">', markdown_img)
            with open(file_path, 'w') as f:
                # 写入替换后的内容
                f.write(content)

