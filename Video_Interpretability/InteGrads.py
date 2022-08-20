"""
INTEGRATED GRADIENTS for SYNCNET
Using Captum
https://captum.ai/#quickstart

"""
import sys
sys.path.append('/srv/home/dsaha/dubbing/pytorch_speech_features')

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
import matplotlib.cm as cm
import pytorch_speech_features
from captum.attr import IntegratedGradients
from utils.misc import savepair
from scipy.ndimage.filters import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

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
        self.__Sig = IntegratedGradients(self.__S__)

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

        im = np.stack(images,axis=3)
        im = np.expand_dims(im,axis=0)
        im = np.transpose(im,(0,3,4,1,2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        
        # ========== ==========
        # Load audio
        # ========== ==========
        
        sample_rate, audio = wavfile.read(os.path.join(opt.tmp_dir,opt.reference,'audio.wav'))
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

        # Since in the changed implementation, cct (audio feature) is the actual audio, we need to compute the mfcc
        mfcc = pytorch_speech_features.mfcc(cct, sample_rate).T
        cc = torch.unsqueeze(torch.unsqueeze(mfcc, dim=0), dim=0)
        cct = cc.float()
        
        lastframe = min_length-5
        im_feat = []
        cc_feat = []
        delta_list = []

        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):

            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            cc_batch = [ cct[:,:,:,vframe*4:vframe*4+20] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            cc_in = torch.cat(cc_batch,0)
            # im_out  = self.__S__.forward_lip(im_in);
            # cc_out  = self.__S__.forward_aud(cc_in)
            # Obtain Integrated Gradients
            (im_in_attr, cc_in_attr), delta = self.__Sig.attribute((im_in, cc_in), n_steps=100, 
                                                                   return_convergence_delta=True, internal_batch_size=10)
            
            # Save in list
            im_feat.append(im_in_attr.detach().cpu().numpy())
            cc_feat.append(cc_in_attr.detach().cpu().numpy())
            delta_list.append(delta.detach().cpu().numpy())

        # im_feat = torch.cat(im_feat,0)
        # cc_feat = torch.cat(cc_feat,0)
        im_feat = np.concatenate(im_feat, axis=0)
        cc_feat = np.concatenate(cc_feat, axis=0)
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
        print('AV offset: \t%d \nMin dist: \t%.3f\nConfidence: \t%.3f' % (offset,minval,conf))

        dists_npy = np.array([ dist.numpy() for dist in dists ])
        return offset.numpy(), minval.numpy(), conf.numpy()

    def evaluate(self, opt, videofile):
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
        isvideo = np.squeeze(np.absolute(imtv), axis=0)
        norm_video = (isvideo - np.min(isvideo))/(np.max(isvideo) - np.min(isvideo))
        # norm_video = np.log(1+norm_video) / np.log(2) # Log-scale normalised image
        isvideo = (max_lvl - min_lvl)*norm_video + min_lvl
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
        
    def savevidoverlay(self, video, mask, f, msg):
        th = 5
        indicator = np.repeat(np.mean(mask, axis = -1)[:, :, :, np.newaxis], 3, axis=-1)
        scaled_mask = (gaussian_filter(mask, sigma=5))
        scaled_mask = (scaled_mask*(np.array([255/x.max() for x in scaled_mask]).reshape(-1,1,1,1))).astype(np.uint8)
        heatmaps = np.zeros_like(mask)
        for i, sm in enumerate(scaled_mask):
            heatmaps[i] = cv2.applyColorMap(sm, cv2.COLORMAP_JET) * (indicator[i] > th)
            heatmaps[i] = cv2.dilate(heatmaps[i], kernel = np.ones((3, 3), np.uint8))
        overlaid = np.zeros_like(heatmaps)
        overlaid = (0.7*video[:-1,:,:,:] + 0.3*heatmaps).astype(np.uint8)
        self.savevid(overlaid, f, msg)

    def savevid(self, video, vidpath, msg):
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
        if m is None:
            m = np.max(np.abs(audio))
        sigf32 = (audio/m).astype(np.float32)
        wavfile.write(audpath, sample_rate, sigf32)
    
    def savespec(self, specpath, audio, sample_rate):
        feat, _ = pytorch_speech_features.fbank(audio.cuda(), sample_rate)
        feat = np.log(1+abs(feat.T.cpu().numpy()))
        ax = sns.heatmap(feat)
        plt.savefig(specpath)
        plt.clf()
        
    def saveaudgrad(self, gradpath, distpath, grad):
        grad = ema(np.array(grad), 100)
        ema_grad, gi = streak(np.uint16(grad > np.mean(grad)+2*np.std(grad)), 20)
        cctgradmat = grad[:100*int(len(grad)/100)].reshape(100, -1)
        ax = sns.heatmap(cctgradmat)
        plt.savefig(gradpath)
        plt.clf()
        # print("Len = {}".format(len(grad)))
        # print("Max = {}".format(np.max(grad)))
        # print("Mean = {}".format(np.mean(grad)))
        # print("Min = {}".format(np.min(grad)))
        # print("Std-Dev = {}".format(np.std(grad)))
        # print("Streak indices = {}".format(gi))
        ema_grad_sq = ema_grad[:100*int(len(grad)/100)].reshape(100, -1)
        ax = sns.heatmap(ema_grad_sq)
        plt.savefig(distpath)
        plt.clf()
        return gi

    def saveaudiograd(self, gradpath, grad):
        grad = np.transpose(np.absolute(grad), (1,0))
        ax = sns.heatmap(grad)
        plt.savefig(gradpath)
        plt.clf()
    
    def save(self, opt, audio, video, vidpath, sample_rate, msg = 'NoMsg'):
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
            
    def test(self, opt, dataset):
        """
        Integrated Grads - On Merkel Dataset
        """
        if not os.path.exists(opt.attack_dir):
            os.makedirs(opt.attack_dir)
        
        # Loop over all examples in test set
        for i in range(0, len(dataset)):
            try:
                # Send the data and label to the device
                videofile = dataset[i]
                attack_folder = os.path.join(opt.attack_dir, os.path.basename(videofile).strip('.avi'))
                if os.path.exists(attack_folder):
                    rmtree(attack_folder)

                # For Saving
                reference = os.path.basename(videofile).strip('.avi')
                opt.reference = reference

                if not videofile.endswith('avi'):
                    videofile = os.path.join(videofile, os.path.basename(videofile)+'.avi')

                # Preprocess
                imtv, cct, min_length, sample_rate = self.preprocess(opt, videofile)
                if opt.shift != 0:
                    print("Shifting signals by {} seconds".format(opt.shift))
                    vshft = int(opt.shift*25)
                    ashft = vshft*640 # 16KHz/25fps
                    assert vshft < imtv.shape[2] and ashft < cct.shape[0]
                    if opt.shift > 0:
                        imtv = imtv[:,:,vshft:,:,:]
                        cct = cct[ashft:]
                    else:
                        imtv = imtv[:,:,:vshft,:,:]
                        cct = cct[:ashft]
                    min_length = min_length - np.abs(vshft)

                # Set requires_grad attribute of tensor. Important for Attack
                imtv = Variable(imtv.data.cuda(), requires_grad = True)
                cct = Variable(cct.data.cuda(), requires_grad = True)

                # Forward pass the data through the model and get Integrated Gradients
                im_feat, cc_feat, tS = self.gen_feat(opt, imtv, cct, min_length, sample_rate)

                # Save Grad Video
                im_feat_accumulate = np.zeros(imtv.shape)
                num_acc = np.zeros(imtv.shape[2])
                for i in range(im_feat.shape[0]):
                    for j in range(im_feat.shape[2]):
                        im_feat_accumulate[0, :, i+j, :, :] += im_feat[i, :, j, :, :]
                        num_acc[i+j] += 1
                imgrads = np.divide(im_feat_accumulate[:, :, :-1, :, :], np.expand_dims(num_acc[:-1], axis=(0, 1, -1, -2)))
                print("Saving Gradients for Video")
                f = "{}/{}/Grad_video.avi".format(opt.attack_dir, os.path.basename(videofile).strip('.avi'))
                grad_video = self.postprocess_grad(imgrads)
                self.savevid(grad_video, f, "")

                # Save original
                f = "{}/{}/Original.avi".format(opt.attack_dir, os.path.basename(videofile).strip('.avi'))
                audio, video = self.postprocess(opt, imtv, cct)
                self.save(opt, audio, video, f, sample_rate)
                f = "{}/{}/Overlay.avi".format(opt.attack_dir, os.path.basename(videofile).strip('.avi'))
                self.savevidoverlay(video, grad_video, f, "")

                # Save Grad Audio
                cc_feat_accumulate = np.zeros((cc_feat.shape[0]*4 + cc_feat.shape[3], cc_feat.shape[2]))
                cc_num_acc = np.zeros(cc_feat.shape[0]*4 + cc_feat.shape[3])
                for i in range(cc_feat.shape[0]):
                    for j in range(cc_feat.shape[3]):
                        cc_feat_accumulate[4*i+j, :] += cc_feat[i, 0, :, j]
                        cc_num_acc[4*i+j] += 1
                ccgrads = np.divide(cc_feat_accumulate[:-4, :], np.expand_dims(cc_num_acc[:-4], axis=1))
                f1 = "{}/{}/Grad_audio.png".format(opt.attack_dir, os.path.basename(videofile).strip('.avi'))
                print("Saving Gradients for Audio")
                self.saveaudiograd(f1, ccgrads)
                amps = np.max(ccgrads, axis = 1)
                plt.plot(amps)
                fx = "{}/{}/Grad_audio_max.png".format(opt.attack_dir, os.path.basename(videofile).strip('.avi'))
                plt.savefig(fx)
                plt.clf()
                print("Saved in {}".format(os.path.join(opt.attack_dir, os.path.basename(videofile).strip('.avi'))))
            except Exception as e:
                print(e)
                pass
        return None

def evaluate(opt):
    if opt.filename != '':
        filename = opt.filename
        print("Evaluating {}".format(filename))
        reference = filename.split('/')[-1].split('.')[0]
        opt.reference = reference
        offset, dist, conf = opt.s.evaluate(opt,videofile=filename)
        print("Offset", offset, "\nDist", dist, "\nConf", conf)
    else:
        with open(opt.filename_list, 'r') as f:
            data = [x.strip() for x in f.readlines()]
        dists = []
        for filename in data:
            print(filename)
            reference = filename.split('/')[-1].split('.')[0]
            opt.reference = reference
            offset, dist, conf = opt.s.evaluate(opt,videofile=filename)
            print("Offset", offset, "\nDist", dist, "\nConf", conf)
            dists.append(dist)
        print("Average = {}".format(np.mean(dists)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "SyncNet Evaluation")
    parser.add_argument('--model', type=str, default="syncnet_evaluation/data/syncnet_v2.model", help='')
    parser.add_argument('--batch_size', type=int, default='20', help='')
    parser.add_argument('--vshift', type=int, default='15', help='')
    parser.add_argument('--out_dir', type=str, default='', help='')
    parser.add_argument('--data_dir', type=str, default='', help='Can be a file name or a folder containing several other folders')
    parser.add_argument('--filename', type=str, default='', help='Name of the video file')
    parser.add_argument('--filename_list', type=str, default='', help='Path to file with video paths')
    parser.add_argument('--alpha', type=float, default=1e5, help='Loss Weighing Term')
    parser.add_argument('--epsilon', type=float, default=0.2, help='Max perturbation magnitude (>0)')
    parser.add_argument('--threshold', type=float, default=2, help='Threshold increase in distance to classify example as fooled')
    parser.add_argument('--evaluation_mode', help='If enabled, runs evaluation on filename instead', action="store_true")
    parser.add_argument('--maxrange', type=int, default='20', help='If using Dataset, range till which to run')
    parser.add_argument('--descent', help='If enabled, runs gradient descent on loss', action="store_true")
    parser.add_argument('--shift', type=float, default='0', help='Shift in secs')

    opt = parser.parse_args();

    setattr(opt,'avi_dir',os.path.join(opt.out_dir,'pyavi'))
    setattr(opt,'tmp_dir',os.path.join(opt.out_dir,'pytmp'))
    setattr(opt,'work_dir',os.path.join(opt.out_dir,'pywork'))
    setattr(opt,'crop_dir',os.path.join(opt.out_dir,'pycrop'))
    setattr(opt,'attack_dir',os.path.join(opt.out_dir,'pyattack'))

    opt.s = SyncNetInstance();

    opt.s.loadParameters(opt.model);
    print("Model %s loaded."%opt.model);
    
    if opt.evaluation_mode:
        # Proper Evaluation
        evaluate(opt)
    else:
        if opt.filename != '':
            data = [opt.filename]
        elif opt.filename_list != '':
            with open(opt.filename_list, 'r') as f:
                data = [x.strip() for x in f.readlines()]
        else:
            # Dataset
            p = '../visualHeroes/german_nb/configs/final_new_merkel_configs/merkel_config_refined_old.json'
            dataset = Syncnet_Dataset(config_path = p)
            maxlim = min(len(dataset), opt.maxrange)
            data = []
            for i in range(maxlim):
                data.append(dataset[i][2])
                
        # FGSM Attack
        opt.s.test(opt, data)