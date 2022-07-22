import cv2

def split_video(input_video, output_video):
    video_caputre = cv2.VideoCapture(input_video)

    # get video parameters
    fps = video_caputre.get(cv2.CAP_PROP_FPS)
    width = video_caputre.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_caputre.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("fps:", fps)
    print("width:", width)
    print("height:", height)

    # 定义截取尺寸,后面定义的每帧的h和w要于此一致，否则视频无法播放
    split_width = int(width / 2)
    split_height = int(height)
    size = (split_width, split_height)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # 创建视频写入对象
    videp_write = cv2.VideoWriter(output_video, fourcc, fps, size)

    print('Start!!!')
    # 读取视频帧
    success, frame_src = video_caputre.read()  # (960, 2560, 3)  # (height, width, channel)
    while success and not cv2.waitKey(1) == 27:  # 读完退出或者按下 esc 退出

        # [width, height] 要与上面定义的size参数一致，注意参数的位置
        # frame_target = frame_src[0:split_height, 0:split_width]  # (split_height, split_width)
        frame_target = frame_src[0:split_height, split_width:int(width)]  # (split_height, split_width)
        # 写入视频文件
        videp_write.write(frame_target)
        # 不断读取
        success, frame_src = video_caputre.read()

    print("Finished!!!")
    video_caputre.release()

if __name__ == '__main__':
    input_file = 'DJI_0078.MP4'
    output_file = 'video_file/demo1.MP4'
    split_video(input_video=input_file, output_video=output_file)
