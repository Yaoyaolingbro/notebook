
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
