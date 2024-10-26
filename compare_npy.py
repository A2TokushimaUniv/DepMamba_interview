import numpy as np

dvlog_audio = np.load("./datasets/dvlog/0/0_acoustic.npy")
lmvd_audio = np.load("./datasets/lmvd/audio/001.npy")
print(f'dvlog_audio: {dvlog_audio.shape}')
print(f'lmvd_audio: {lmvd_audio.shape}')


dvlog_video = np.load("./datasets/dvlog/0/0_visual.npy")
lmvd_video = np.load("./datasets/lmvd/visual/001_visual.npy")
print(f'dvlog_video: {dvlog_video.shape}')
print(f'lmvd_video: {lmvd_video.shape}')