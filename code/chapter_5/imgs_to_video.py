"""
This py file is used to make videos for the anim.cu file in the chapter_5.
"""
import cv2
import os

def images_to_video(image_folder, output_video_file, fps=30):
    """
    将指定文件夹中的图片整理成视频并保存。

    :param image_folder: 包含帧图片的文件夹路径。
    :param output_video_file: 保存视频的路径，例如 "output.mp4"。
    :param fps: 视频的帧率 (Frames Per Second)。
    """
    # 获取图片列表，并按文件名排序
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # 按图片文件名中的帧编号排序

    # 如果没有图片，则退出
    if not images:
        print("No images found in folder:", image_folder)
        return

    # 读取第一张图片以获取宽高信息
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 定义视频编码器（使用 MP4 编码器）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 保存为 MP4 格式
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # 遍历所有图片并写入视频
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # 释放视频对象
    video.release()
    print(f"Video saved to {output_video_file}")

# 使用示例
image_folder = "./imgs"  # 替换为你的图片文件夹路径
output_video_file = "anim.mp4"  # 替换为你希望保存的视频路径
images_to_video(image_folder, output_video_file, fps=30)
