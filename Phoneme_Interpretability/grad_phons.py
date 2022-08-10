"""
Code to save high-gradient regions in the audio by simply backpropagating on the audio sample.

"""
import sys
sys.path.append('/srv/home/dsaha/dubbing/pytorch_speech_features')
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
from syncnet_evaluation.SyncNetModel import *
from utils.dataset import Syncnet_Dataset
import time, pdb, argparse, subprocess, os, math, glob
import cv2
import python_speech_features
from scipy import signal
from scipy.io import wavfile
from shutil import rmtree
from librosa.feature.inverse import mfcc_to_audio
import seaborn as sns
import matplotlib.pylab as plt
import pytorch_speech_features
from utils.math_utils import ema, streak

# Params
use_cuda=True

# ==================== Get OFFSET ====================

def calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift*2+1
    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))
    dists = []
    for i in range(0,len(feat1)):
        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))
    return dists

class SyncNetInstance(torch.nn.Module):

    def __init__(self, dropout = 0, num_layers_in_fc_layers = 1024):
        super(SyncNetInstance, self).__init__();

        self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers).cuda();

    def preprocess(self, opt, videofile):
        self.__S__.eval();

        # ========== ==========
        # Convert files
        # ========== ==========

        if os.path.exists(os.path.join(opt.tmp_dir,opt.reference)):
            rmtree(os.path.join(opt.tmp_dir,opt.reference))
        os.makedirs(os.path.join(opt.tmp_dir,opt.reference))

        command = ("ffmpeg -y -i %s -threads 1 -f image2 %s -hide_banner -loglevel error" % (videofile,os.path.join(opt.tmp_dir,opt.reference,'%06d.jpg')))
        output = subprocess.call(command, shell=True, stdout=None)

        command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s -hide_banner -loglevel error" % (videofile,os.path.join(opt.tmp_dir,opt.reference,'audio.wav')))
        output = subprocess.call(command, shell=True, stdout=None)

        # ========== ==========
        # Load video
        # ========== ==========

        images = []
        flist = glob.glob(os.path.join(opt.tmp_dir,opt.reference,'*.jpg'))
        flist.sort()
        for fname in flist:
            images.append(cv2.imread(fname))
        if opt.trim_t1 and opt.trim_t2:
            print("Cropping between {} and {}".format(opt.trim_t1, opt.trim_t2))
            images = images[int(25*opt.trim_t1): int(25*opt.trim_t2)]
        
        # Tensor from the cropped video segment
        im = np.stack(images,axis=3)
        im = np.expand_dims(im,axis=0)
        im = np.transpose(im,(0,3,4,1,2))
        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        
        # ========== ==========
        # Load audio
        # ========== ==========
        
        # Create a tensor from the actual audio segment
        sample_rate, audio = wavfile.read(os.path.join(opt.tmp_dir,opt.reference,'audio.wav'))
        if opt.trim_t1 and opt.trim_t2:
            print("Cropping between {} and {}".format(opt.trim_t1, opt.trim_t2))
            audio = audio[int(sample_rate*opt.trim_t1): int(sample_rate*opt.trim_t2)]
        audio_tensor = torch.autograd.Variable(torch.from_numpy(audio).double())
        audio_tensor.requires_grad_()

        # ========== ==========
        # Check audio and video input length
        # ========== ==========

        if (float(len(audio))/16000) != (float(len(images))/25) :
            print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different."%(float(len(audio))/16000,float(len(images))/25))

        min_length = min(len(images),math.floor(len(audio)/640))
        return imtv, audio_tensor, min_length, sample_rate

    def gen_feat(self, opt, imtv, cct, min_length, sample_rate):
        # ========== ==========
        # Generate video and audio feats
        # ========== ==========

        # Computing audio features
        # Since in the changed implementation, cct (audio feature) is the actual audio, we need to compute the mfcc
        # To allow direct gradient computation on the audio signal, we use pytorch-speech-features.
        mfcc = pytorch_speech_features.mfcc(cct, sample_rate).T
        cc = torch.unsqueeze(torch.unsqueeze(mfcc, dim=0), dim=0)
        cct = cc.float()
        
        # Computing video features
        lastframe = min_length-5
        im_feat = []
        cc_feat = []

        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):

            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.__S__.forward_lip(im_in);
            im_feat.append(im_out)

            cc_batch = [ cct[:,:,:,vframe*4:vframe*4+20] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            cc_in = torch.cat(cc_batch,0)
            cc_out  = self.__S__.forward_aud(cc_in)
            cc_feat.append(cc_out)

        im_feat = torch.cat(im_feat,0)
        cc_feat = torch.cat(cc_feat,0) # L x 1024 (output dim)
        return im_feat, cc_feat, tS

    def get_mdist(self, opt, im_feat, cc_feat):
        # ========== ==========
        # Compute Distance
        # ========== ==========
        dists = calc_pdist(im_feat,cc_feat,vshift=opt.vshift)
        mdist = torch.mean(torch.stack(dists,1),1)
        return mdist
    
    def get_offset(self, opt, im_feat, cc_feat, tS):
        # ========== ==========
        # Compute offset
        # ========== ==========
        im_feat = im_feat.clone().cpu().detach()
        cc_feat = cc_feat.clone().cpu().detach()
        
        print('Compute time %.3f sec.' % (time.time()-tS))

        dists = calc_pdist(im_feat,cc_feat,vshift=opt.vshift)
        mdist = torch.mean(torch.stack(dists,1),1)

        minval, minidx = torch.min(mdist,0)

        offset = opt.vshift-minidx
        conf   = torch.median(mdist) - minval

        fdist   = np.stack([dist[minidx].numpy() for dist in dists])
        fconf   = torch.median(mdist).numpy() - fdist
        fconfm  = signal.medfilt(fconf,kernel_size=9)

        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        # print('Framewise conf: ')
        # print(fconfm)
        print('AV offset: \t%d \nMin dist: \t%.3f\nConfidence: \t%.3f' % (offset,minval,conf))

        dists_npy = np.array([ dist.numpy() for dist in dists ])
        return offset.numpy(), minval.numpy(), conf.numpy()

    def evaluate(self, opt, videofile):
        # Full evaluation (All steps)
        imtv, cct, min_length, sample_rate = self.preprocess(opt, videofile)
        imtv = Variable(imtv.data.cuda(), requires_grad=True)
        cct = Variable(cct.data.cuda(), requires_grad=True)
        im_feat, cc_feat, tS = self.gen_feat(opt, imtv, cct, min_length, sample_rate)
        offset, minval, conf = self.get_offset(opt, im_feat, cc_feat, tS)
        return offset, minval, conf
    
    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage);
        self_state = self.__S__.state_dict();
        for name, param in loaded_state.items():
            self_state[name].copy_(param);

    # ========== ==========
    # Postprocessing Functions
    # ========== ==========
    def postprocess_video(self, opt, imtv):
        # Video
        isvideo = torch.squeeze(imtv.clone().cpu().detach(), 0).numpy()
        video = np.transpose(isvideo,(1,2,3,0)).astype('uint8')
        return video
    
    def postprocess_audio(self, opt, cct):
        # Audio
        audio = torch.squeeze(torch.squeeze(cct.clone().cpu().detach(), 0), 0).numpy()
        return audio
    
    def postprocess_grad(self, imtv, min_lvl = 0, max_lvl = 255):
        # Video
        isvideo = np.absolute(torch.squeeze(imtv.clone().cpu().detach(), 0).numpy())
        isvideo = (max_lvl - min_lvl)*(isvideo - np.min(isvideo))/(np.max(isvideo) - np.min(isvideo) + 1e-9) + min_lvl
        video = np.transpose(isvideo,(1,2,3,0)).astype('uint8')
        return video

    def postprocess_diff(self, imtv, min_lvl = 0, max_lvl = 255):
        # Video
        isvideo = torch.squeeze(imtv.clone().cpu().detach(), 0).numpy()
        isvideo = (max_lvl - min_lvl)*(isvideo - np.min(isvideo))/(np.max(isvideo) - np.min(isvideo) + 1e9) + min_lvl
        video = np.transpose(isvideo,(1,2,3,0)).astype('uint8')
        return video
    
    def postprocess(self, opt, imtv, cct):
        audio = self.postprocess_audio(opt, cct)
        video = self.postprocess_video(opt, imtv)
        return audio, video

    # ========== ==========
    # Saving utily functions
    # ========== ==========
    def savevid(self, video, vidpath, msg):
        # Saving video
        # Make Dir
        if not os.path.exists(os.path.dirname(vidpath)):
            os.makedirs(os.path.dirname(vidpath))
        
        # Writing Video
        # Dims
        frame_height = int(video.shape[1])
        frame_width = int(video.shape[2])
        frame_channel = int(video.shape[3])
        size = (frame_width, frame_height)
        # Below VideoWriter object
        result = cv2.VideoWriter(vidpath, 
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 25, size, True)
        for i in range(video.shape[0]):
            frame = video[i, :, :, :].copy() 
            
            if msg != "":
                # Write on Video
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 
                            msg, 
                            (0, 20), 
                            font, 0.4, 
                            (0, 255, 255), 
                            1, 
                            cv2.LINE_4)
            result.write(frame)
  
        # When everything done, release 
        result.release()
    
    def saveaud(self, audpath, audio, sample_rate, m = None):
        # Saving audio
        if m is None:
            m = np.max(np.abs(audio))
        sigf32 = (audio/m).astype(np.float32)
        wavfile.write(audpath, sample_rate, sigf32)
    
    def savespec(self, specpath, audio, sample_rate):
        # Saving spectrograms
        feat, _ = pytorch_speech_features.fbank(audio.cuda(), sample_rate)
        feat = np.log(1+abs(feat.T.cpu().numpy()))
        ax = sns.heatmap(feat)
        plt.savefig(specpath)
        plt.clf()
        
    def saveaudgrad(self, gradpath, distpath, grad):
        # Saving audio gradient
        grad = np.absolute(grad)
        grad = ema(np.array(grad), 100)
        ema_grad, gi = streak(np.uint16(grad > np.mean(grad)+2*np.std(grad)), 20)
        cctgradmat = grad[:100*int(len(grad)/100)].reshape(100, -1)
        ax = sns.heatmap(cctgradmat)
        plt.savefig(gradpath)
        plt.clf()
        print("Len = {}".format(len(grad)))
        print("Max = {}".format(np.max(grad)))
        print("Mean = {}".format(np.mean(grad)))
        print("Min = {}".format(np.min(grad)))
        print("Std-Dev = {}".format(np.std(grad)))
        print("Streak indices = {}".format(gi))
        ema_grad_sq = ema_grad[:100*int(len(grad)/100)].reshape(100, -1)
        ax = sns.heatmap(ema_grad_sq)
        plt.savefig(distpath)
        print("Saved in {}".format(distpath))
        plt.clf()
        return gi
    
    def calcaudiosegs(self, grad):
        # Calculate segments
        grad = np.absolute(grad)
        grad = ema(np.array(grad), 100)
        ema_grad, gi = streak(np.uint16(grad > np.mean(grad)+2*np.std(grad)), 20)
        return gi

    def save(self, opt, audio, video, vidpath, sample_rate, msg = 'NoMsg'):
        # Save all
        # Make Dir
        if not os.path.exists(os.path.dirname(vidpath)):
            os.makedirs(os.path.dirname(vidpath))
        
        # Writing Video
        self.savevid(video, vidpath, msg)
        
        # Writing Audio
        audpath = vidpath.strip('avi')+'wav'
        self.saveaud(audpath, audio, sample_rate)
        
        # Writing Spec
        specpath = vidpath.strip('avi')+'png'
        self.savespec(specpath, torch.tensor(audio), sample_rate)
        
        # Combine
        newvidpath = vidpath.strip('.avi')+'_combined.avi'
        command = ("ffmpeg -i %s -i %s -c:v copy -c:a aac %s -hide_banner -loglevel error" % (vidpath, audpath, newvidpath))
        output = subprocess.call(command, shell=True, stdout=None)

    # ========== ==========
    # Main test function
    # ========== ==========
    def test(self, opt, dataset):
        """
        Saving high-gradient audio segments
        """
        # What to perturb
        if not opt.perturb_audio and not opt.perturb_video:
            raise ValueError("Target Modality not specified")

        # Gradient Descent
        if opt.descent:
            opt.epsilon = -opt.epsilon
        
        # Accuracy counter
        correct = 0
        done = 0
        all_del_dist = []
        all_grad_intervals = []
        
        # Loop over all examples in test set
        for i in range(0, len(dataset)):
            try:
                # Send the data and label to the device
                videofile = dataset[i]

                # For Saving
                reference = os.path.basename(videofile).strip('.avi')
                opt.reference = reference

                if not videofile.endswith('avi') and not videofile.endswith('avi'):
                    videofile = os.path.join(videofile, os.path.basename(videofile)+'.avi')

                # Preprocess
                imtv, cct, min_length, sample_rate = self.preprocess(opt, videofile)

                # Set requires_grad attribute of tensor. Important for Attack
                imtv = Variable(imtv.data.cuda(), requires_grad=opt.perturb_video)
                cct = Variable(cct.data.cuda(), requires_grad=opt.perturb_audio)

                # Forward pass the data through the model
                im_feat, cc_feat, tS = self.gen_feat(opt, imtv, cct, min_length, sample_rate)

                # Distance
                mdist = torch.mean(self.get_mdist(opt, im_feat, cc_feat))

                # Maximise dist - Since we do gradient ascent, we MAXIMISE the loss defined below
                loss = opt.alpha * mdist

                # Zero all existing gradients
                self.__S__.zero_grad()

                # Calculate gradients of model in backward pass
                loss.backward()

                # Collect datagrad
                cct_grad = cct.grad
                imtv_grad = imtv.grad

                # Call FGSM Attack
                if opt.perturb_audio:
                    perturbed_cct = fgsm_attack(cct, opt.epsilon, cct_grad, None, None)
                else:
                    perturbed_cct = cct
                if opt.perturb_video:
                    perturbed_imtv = fgsm_attack(imtv, opt.epsilon, imtv_grad)
                else:
                    perturbed_imtv = imtv

                if not cct.grad is None:
                    grad_intervals = self.calcaudiosegs(cct.grad.detach().cpu())
                    t_offset = 0
                    if opt.trim_t1:
                        t_offset = opt.trim_t1
                    time_interval = [list(t_offset + g/16000) for g in grad_intervals]
                    all_grad_intervals.append(time_interval)

            except Exception as e:
                print(e)

        # Return the high-gradient time-segments
        return all_grad_intervals
            
# FGSM attack code
def fgsm_attack(image, epsilon, data_grad, min_lvl = 0, max_lvl = 255):
    use_audio = (min_lvl is None or max_lvl is None)
    if use_audio:
        epsilon = epsilon * torch.max(torch.absolute(image))/255
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,255] range
    if min_lvl is not None and max_lvl is not None:
        perturbed_image = torch.clamp(perturbed_image, min_lvl, max_lvl)
    # Return the perturbed image
    return perturbed_image

# Dict --> Object
class MyObject:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)

# Load Model
def initialize():
    # Default params
    opt = {'model': '../syncnet_evaluation/data/syncnet_v2.model', 
           'batch_size': 20, 'vshift': 15, 
           'out_dir': '', 
           'data_dir': '', 
           'filename': None, 
           'filename_list': '', 
           'alpha': 100000.0, 
           'epsilon': 0.2, 
           'threshold': 2, 
           'perturb_audio': True, 
           'perturb_video': False, 
           'evaluation_mode': False, 
           'maxrange': 20, 
           'descent': False, 
           'shift': 0.0, 
           'trim_t1': None, 
           'trim_t2': None}
    opt = MyObject(opt)
    opt.s = SyncNetInstance();
    opt.s.loadParameters(opt.model);
    print("Model %s loaded."%opt.model);
    
    setattr(opt,'avi_dir',os.path.join(opt.out_dir,'pyavi'))
    setattr(opt,'tmp_dir',os.path.join(opt.out_dir,'pytmp'))
    setattr(opt,'work_dir',os.path.join(opt.out_dir,'pywork'))
    setattr(opt,'crop_dir',os.path.join(opt.out_dir,'pycrop'))
    setattr(opt,'attack_dir',os.path.join(opt.out_dir,'pyattack'))
    
    return opt 

def run(opt, filename, t1, t2):
    # Modify the attack .py script to collect high-gradient video segments
    opt.filename = filename
    
    if opt.filename != '':
        data = [opt.filename]
    elif opt.filename_list != '':
        with open(opt.filename_list, 'r') as f:
            data = [x.strip() for x in f.readlines()]
    else:
        raise Exception("Data source not specified")
            
    # FGSM Attack
    grad_intervals = opt.s.test(opt, data)
    return grad_intervals