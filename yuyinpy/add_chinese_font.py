import os
import shutil
from pathlib import Path

# 可能的中文字体路径
windows_font_paths = [
    "C:/Windows/Fonts/simhei.ttf",  # 黑体
    "C:/Windows/Fonts/simsun.ttc",  # 宋体
    "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
    "C:/Windows/Fonts/simkai.ttf",  # 楷体
]

# 创建fonts目录
fonts_dir = Path("./fonts")
fonts_dir.mkdir(exist_ok=True)

# 尝试复制字体
success = False
for font_path in windows_font_paths:
    try:
        if os.path.exists(font_path):
            dest_path = fonts_dir / Path(font_path).name
            shutil.copy2(font_path, dest_path)
            print(f"成功复制字体: {font_path} -> {dest_path}")
            success = True
    except Exception as e:
        print(f"复制字体失败: {font_path}, 错误: {e}")

if not success:
    print("警告: 未能复制任何中文字体，GUI可能无法正确显示中文。")
    print("请手动将中文字体文件复制到 ./fonts 目录。") 