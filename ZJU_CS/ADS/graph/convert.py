import os

# 获取当前目录下所有文件
for filename in os.listdir("."):
    # 如果文件名符合要求
    if filename.endswith(".png") and "ads_hw_" in filename:
        # 将 . 替换为 _
        new_filename = filename.replace(".", "_", 1)  # 只替换第一个点
        # 重命名文件
        os.rename(filename, new_filename)
        print(f"Renamed: {filename} -> {new_filename}")