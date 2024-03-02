import os
import re

def convert_html_to_markdown(html_string):
    # 使用正则表达式匹配 HTML 图片标签
    pattern = r'<div style="text-align:center;">\s*<img src="([^"]+)" alt="([^"]+)" style="margin: 0 auto; zoom: 80%;" />\s*</div>'
    matches = re.finditer(pattern, html_string)

    # 将匹配到的 HTML 图片标签转换为 Markdown 语法
    markdown_images = []
    for match in matches:
        src = match.group(1)
        alt = match.group(2)
        markdown_image = f"![{alt}]({src})"
        markdown_images.append(markdown_image)

    # 将原始 HTML 图片标签替换为 Markdown 图片语法
    markdown_string = re.sub(pattern, '\n'.join(markdown_images), html_string)
    return markdown_string

def process_markdown_files_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                markdown_content = file.read()
            
            # 调用转换函数，获取转换后的 Markdown 内容
            converted_markdown_content = convert_html_to_markdown(markdown_content)
            
            # 将转换后的内容写回原始文件
            with open(filepath, "w") as file:
                file.write(converted_markdown_content)
            print(f"Converted {filename}")

# 将需要转换的文件夹路径传递给函数
process_markdown_files_in_directory("docs")
