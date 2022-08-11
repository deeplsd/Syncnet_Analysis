import cv2
import python_speech_features
from scipy import signal
from scipy.io import wavfile
from shutil import rmtree
from librosa.feature.inverse import mfcc_to_audio
import seaborn as sns
import numpy as np
import matplotlib.pylab as plt
import pytorch_speech_features

def savepair(vg, ag, fig_width = 1024, img_width = 64):
    La, _ = ag.shape
    _, _, Lv, _, _ = vg.shape
    assert La == Lv*4
    nacopy = 1
    while Lv*nacopy < img_width:
        nacopy *= 2
    print(nacopy)
    pix = 4*nacopy
    sample_per = img_width/pix
    print(sample_per)
    sampled_vg = np.transpose(vg[0, :, ::int(sample_per), :, :], (1,2,3,0))
    print(sampled_vg.shape)