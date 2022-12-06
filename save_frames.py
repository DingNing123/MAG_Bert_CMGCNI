# 把源视频分解为帧图片


video_path = 'mmsd_raw_data/utterances_final/'
frame_path = 'mmsd_raw_data/Processed/video/Frames/'
from glob import glob
videos = glob(video_path+'*')
print(len(videos),type(videos))
train_ids = ['1_70','1_276']
test_ids = ['1_60','1_80']
import os
# for video in videos[:2]:
for video in videos[:]:
    video = video.split('/')[2].split('.')[0]
    if video in train_ids + test_ids:
        dirName = frame_path + video
        # Create target Directory if don't exist
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ")
        else:
            print("Directory " , dirName ,  " already exists")
        input_mp4 = video_path + video + ".mp4"
        # os.environ['path']="/Users/mac/anaconda3/envs/t18/bin"
        ffmpeg = '/Users/mac/anaconda3/envs/t18/bin/ffmpeg'
        cmd = "{} -i {} -vf fps=1 {}/%5d.jpg".format(ffmpeg,input_mp4,dirName)
        print(cmd)
        os.system(cmd)
        # print(os.getenv('path'))
        

# ffmpeg -i mmsd_raw_data/utterances_final/1_60.mp4 -vf fps=5 mmsd_raw_data/Processed/video/Frames/1_60/%5d.jpg
# fps frame per seconde 
# 5 is too big ,should be 1 

