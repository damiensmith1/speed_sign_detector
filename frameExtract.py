from moviepy.editor import VideoFileClip
import os

def frameExtract(movie, times, imagedir):
    if not os.path.exists(imagedir):
        os.makedirs(imagedir)

    clip = VideoFileClip(movie)

    for t in times:
        imgpath = os.path.join(imagedir, '{}.png'.format(int(t * clip.fps)))
        clip.save_frame(imgpath, t)

movie = 'testVideo.mp4'
imagedir = './frames'
clip = VideoFileClip(movie)
times = []
for i, frame in enumerate(clip.iter_frames()):
    if i % 5 == 0: #extract every 5th frame
        times.append(i / clip.fps)

frameExtract(movie, times, imagedir)
