import cv2
import os
import re


# 获取图片列表
# 自定义排序函数
def sorted_nicely(l):
    """Human-friendly sorting function"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def Ising_Video(J):
    # 图片所在文件夹路径
    image_folder = 'img'

    # 视频文件名
    if J > 0:
        video_name = 'QMC_Anti_Fe_Ising_Video.avi'
    else:
        video_name = 'QMC_Fe_Ising_Video.avi'

    # 视频帧率（FPS）
    fps = 60
    # 获取图片列表并按照自定义排序规则排序
    images = sorted_nicely([img for img in os.listdir(image_folder) if img.endswith(".png")])

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 创建视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # 关闭视频流
    cv2.destroyAllWindows()
    video.release()

