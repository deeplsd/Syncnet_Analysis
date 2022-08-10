"""
ADVERSARIAL EXAMPLE GENERATION for SYNCNET
Using Fast Gradient Sign Attack
https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

Fool model into thinking video is out-of-sync / in-syncs
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

        im = np.stack(images,axis=3)
        im = np.expand_dims(im,axis=0)
        im = np.transpose(im,(0,3,4,1,2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        
        # ========== ==========
        # Load audio
        # ========== ==========
        
        sample_rate, audio = wavfile.read(os.path.join(opt.tmp_dir,opt.reference,'audio.wav'))
        if opt.trim_t1 and opt.trim_t2:
            print("Cropping between {} and {}".format(opt.trim_t1, opt.trim_t2))
            audio = audio[int(sample_rate*opt.trim_t1): int(sample_rate*opt.trim_t2)]
        audio_tensor = torch.autograd.Variable(torch.from_numpy(audio).double())
        audio_tensor.requires_grad_()
        
        # Since we are backpropagating on the audio-signal, we do not perform feature extraction here
        # mfcc = zip(*python_speech_features.mfcc(audio,sample_rate))
        # mfcc = np.stack([np.array(i) for i in mfcc])
        # cc = np.expand_dims(np.expand_dims(mfcc,axis=0),axis=0)
        # cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

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

    def save(self, opt, audio, video, vidpath, sample_rate, msg = 'NoMsg'):
        # Make Dir
        if not os.path.exists(os.path.dirname(vidpath)):
            os.makedirs(os.path.dirname(vidpath))
        
        # Writing Video
        self.savevid(video, vidpath, msg)
        
        # Writing Audio
        audpath = vidpath[:-4]+'.wav'
        self.saveaud(audpath, audio, sample_rate)
        
        # Writing Spec
        specpath = vidpath[:-4]+'.png'
        self.savespec(specpath, torch.tensor(audio), sample_rate)
        
        # Combine
        newvidpath = vidpath[:-4]+'_combined.mp4'
        command = ("ffmpeg -i %s -i %s -c:v copy -c:a aac %s -hide_banner -loglevel error" % (vidpath, audpath, newvidpath))
        output = subprocess.call(command, shell=True, stdout=None)
            
    def test(self, opt, dataset):
        """
        Fast Gradient Sign Attack - On Merkel Dataset
        """
        if not os.path.exists(opt.attack_dir):
            os.makedirs(opt.attack_dir)
        
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
        
        # Loop over all examples in test set
        for i in range(0, len(dataset)):
            try:
                # Send the data and label to the device
                videofile = dataset[i]
                attack_folder = os.path.join(opt.attack_dir, os.path.basename(videofile).strip('.mp4').strip('.avi'))
                if os.path.exists(attack_folder):
                    rmtree(attack_folder)
                
                # For saving
                reference = os.path.basename(videofile).strip('.mp4').strip('.avi')
                opt.reference = reference

                if not videofile.endswith('mp4') and not videofile.endswith('avi'):
                    videofile = os.path.join(videofile, os.path.basename(videofile)+'.mp4')

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
                imtv = Variable(imtv.data.cuda(), requires_grad=opt.perturb_video)
                cct = Variable(cct.data.cuda(), requires_grad=opt.perturb_audio)

                # Forward pass the data through the model
                im_feat, cc_feat, tS = self.gen_feat(opt, imtv, cct, min_length, sample_rate)

                # Initial Offset
                print("Before Adversarial Attack")
                offset, dist, conf = self.get_offset(opt, im_feat, cc_feat, tS)

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

                # Re-classify the perturbed image
                im_feat, cc_feat, tS = self.gen_feat(opt, perturbed_imtv, perturbed_cct, min_length, sample_rate)
                print('===========')
                print("After Adversarial Attack")
                offset_adv, dist_adv, conf_adv = self.get_offset(opt, im_feat, cc_feat, tS)

                # Check for success
                done += 1
                all_del_dist.append(dist_adv - dist)
                # Condition for correct prediction
                # Ascent - dist_adv not much increased
                # Descent - dist_adv not much decreased
                cond = ((dist_adv - dist) < opt.threshold if not opt.descent else (dist_adv - dist) > -opt.threshold)
                if cond:
                    correct += 1
                    # Special case for smp4ng 0 epsilon examples
                    if (opt.epsilon == 0):
                        print("Fooled")
                else:
                    # Save some adv examples for visualization later
                    print("Fooled")

                # Save Audio
                # Add this as a separate function
                # reduced_cct = np.squeeze(np.squeeze(cct.cpu().detach().numpy(), 0), 0)
                # reduced_perturbed_cct = np.squeeze(np.squeeze(perturbed_cct.cpu().detach().numpy(), 0), 0)
                # plt.figure(figsize = (50, 50))
                # plt.imshow(reduced_perturbed_cct-reduced_cct, cmap='hot', interpolation='nearest')
                # plt.savefig('out.png')
                
                # saving Videos
                f = "{}/{}/Adv_eps_{}.avi".format(opt.attack_dir, reference, str(opt.epsilon).replace('.', '_'))
                audio_1, video_1 = self.postprocess(opt, perturbed_imtv, perturbed_cct)
                msg="Offset={},Dist={},Conf={}".format(round(float(offset_adv), 2),round(float(dist_adv), 2),round(float(conf_adv),2))
                self.save(opt, audio_1, video_1, f, sample_rate, msg)

                f = "{}/{}/Original.avi".format(opt.attack_dir, reference)
                audio_2, video_2 = self.postprocess(opt, imtv, cct)
                msg = "Offset={}, Dist={}, Conf={}".format(round(float(offset), 2), round(float(dist), 2), round(float(conf), 2))
                self.save(opt, audio_2, video_2, f, sample_rate, msg)
                
                if not imtv.grad is None:
                    print("Saving Gradients for Video")
                    f = "{}/{}/Grad_video.avi".format(opt.attack_dir, reference)
                    grad_video = self.postprocess_grad(imtv.grad)
                    self.savevid(grad_video, f, "")

                    f = "{}/{}/Diff_video.avi".format(opt.attack_dir, reference)
                    diff_video = self.postprocess_diff(perturbed_imtv-imtv)
                    self.savevid(diff_video, f, "")
                    print("-------------------------------------------")
                    print("Saved in {}/{}".format(opt.attack_dir, reference))
                    
                if not cct.grad is None:
                    f1 = "{}/{}/Grad_audio.png".format(opt.attack_dir, reference)
                    f2 = "{}/{}/Dist_Grad_audio.png".format(opt.attack_dir, reference)
                    audf = "{}/{}/Diff_audio.wav".format(opt.attack_dir, reference)
                    specf = "{}/{}/Diff_spec.png".format(opt.attack_dir, reference)
                    grad_intervals = self.saveaudgrad(f1, f2, cct.grad.detach().cpu())
                    t_offset = 0
                    if opt.trim_t1:
                        t_offset = opt.trim_t1
                    print("Time Intervals causing large gradients = {}".format(
                        [list(t_offset + g/16000) for g in grad_intervals]))
                    self.savespec(specf, torch.tensor(audio_1-audio_2), sample_rate)
                    self.saveaud(audf, audio_1-audio_2, sample_rate, np.max(np.abs(audio_2)))
                    os.makedirs("{}/{}/Grad_segments/".format(opt.attack_dir, reference))
                    for idx, (i, j) in enumerate(grad_intervals):
                        ap = "{}/{}/Grad_segments/segment_{}.png".format(opt.attack_dir, reference, idx)
                        self.savespec(ap, torch.tensor(audio_2[i:j]), sample_rate)
                    print("-------------------------------------------")
                    print("Saving Gradients for Audio")
                    print("Saved in {}/{}".format(opt.attack_dir, reference))
            except Exception as e:
                print(e)
                pass
        
        # Calculate final accuracy for this epsilon
        final_acc = correct/float(done)
        avg_del_dist = np.mean(all_del_dist)
        print("Epsilon: {}\tTest Accuracy (with threshold = {}) = {} / {} = {}\nAverage increase in distance = {}".format(opt.epsilon,
                                                                                                                      opt.threshold,
                                                                                                                      correct, done,
                                                                                                                      final_acc,
                                                                                                                      avg_del_dist))

        # Return the accuracy and an adversarial example
        return final_acc
            
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
    parser.add_argument('--model', type=str, default="../syncnet_evaluation/data/syncnet_v2.model", help='')
    parser.add_argument('--batch_size', type=int, default='20', help='')
    parser.add_argument('--vshift', type=int, default='15', help='')
    parser.add_argument('--out_dir', type=str, default='', help='')
    parser.add_argument('--data_dir', type=str, default='', help='Can be a file name or a folder containing several other folders')
    parser.add_argument('--filename', type=str, default='', help='Name of the video file')
    parser.add_argument('--filename_list', type=str, default='', help='Path to file with video paths')
    parser.add_argument('--alpha', type=float, default=1e5, help='Loss Weighing Term')
    parser.add_argument('--epsilon', type=float, default=0.2, help='Max perturbation magnitude (>0)')
    parser.add_argument('--threshold', type=float, default=2, help='Threshold increase in distance to classify example as fooled')
    parser.add_argument('--perturb_audio', help='Whether to perturb audio', action="store_true")
    parser.add_argument('--perturb_video', help='Whether to perturb video', action="store_true")
    parser.add_argument('--evaluation_mode', help='If enabled, runs evaluation on filename instead', action="store_true")
    parser.add_argument('--maxrange', type=int, default='20', help='If using Dataset, range till which to run')
    parser.add_argument('--descent', help='If enabled, runs gradient descent on loss', action="store_true")
    parser.add_argument('--shift', type=float, default='0', help='Shift in secs')
    parser.add_argument('--trim_t1', type=float, default=None, help='Time trim start')
    parser.add_argument('--trim_t2', type=float, default=None, help='Time trim end')
    opt = parser.parse_args();

    setattr(opt,'mp4_dir',os.path.join(opt.out_dir,'pymp4'))
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