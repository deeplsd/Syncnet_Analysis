"""
ADVERSARIAL EXAMPLE GENERATION for SYNCNET
Using Basic Iteractive Method (BIM) 
https://www.sciencedirect.com/science/article/pii/S209580991930503X

Fool model into thinking video is out-of-sync / in-syncs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from syncnet_evaluation.SyncNetModel import *
from utils.dataset import Syncnet_Dataset
import time, pdb, argparse, subprocess, os, math, glob
import cv2
from tqdm import tqdm
import python_speech_features
from scipy import signal
from scipy.io import wavfile
from shutil import rmtree
from librosa.feature.inverse import mfcc_to_audio

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

        im = np.stack(images,axis=3)
        im = np.expand_dims(im,axis=0)
        im = np.transpose(im,(0,3,4,1,2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        
        # ========== ==========
        # Load audio
        # ========== ==========
        
        sample_rate, audio = wavfile.read(os.path.join(opt.tmp_dir,opt.reference,'audio.wav'))
        mfcc = zip(*python_speech_features.mfcc(audio,sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])
        
        cc = np.expand_dims(np.expand_dims(mfcc,axis=0),axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

        # ========== ==========
        # Check audio and video input length
        # ========== ==========

        if (float(len(audio))/16000) != (float(len(images))/25) :
            print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different."%(float(len(audio))/16000,float(len(images))/25))

        min_length = min(len(images),math.floor(len(audio)/640))
        return imtv, cct, min_length

    def gen_feat(self, opt, imtv, cct, min_length):
        # ========== ==========
        # Generate video and audio feats
        # ========== ==========

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
        cc_feat = torch.cat(cc_feat,0)
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
        imtv, cct, min_length = self.preprocess(opt, videofile)
        imtv = Variable(imtv.data.cuda(), requires_grad=True)
        cct = Variable(cct.data.cuda(), requires_grad=True)
        im_feat, cc_feat, tS = self.gen_feat(opt, imtv, cct, min_length)
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
        mfcc = torch.squeeze(torch.squeeze(cct.clone().cpu().detach(), 0), 0).numpy()
        audio = mfcc_to_audio(mfcc)        
        return audio
    
    def postprocess_grad(self, imtv, min_lvl = 0, max_lvl = 255):
        # Video
        isvideo = np.absolute(torch.squeeze(imtv.clone().cpu().detach(), 0).numpy())
        isvideo = (max_lvl - min_lvl)*(isvideo - np.min(isvideo))/(np.max(isvideo) - np.min(isvideo)) + min_lvl
        video = np.transpose(isvideo,(1,2,3,0)).astype('uint8')
        return video

    def postprocess_diff(self, imtv, min_lvl = 0, max_lvl = 255):
        # Video
        isvideo = torch.squeeze(imtv.clone().cpu().detach(), 0).numpy()
        isvideo = (max_lvl - min_lvl)*(isvideo - np.min(isvideo))/(np.max(isvideo) - np.min(isvideo)) + min_lvl
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

    def save(self, opt, audio, video, vidpath, sample_rate, msg = 'ABC_NoMsg'):
        # Make Dir
        if not os.path.exists(os.path.dirname(vidpath)):
            os.makedirs(os.path.dirname(vidpath))
        
        # Writing Video
        self.savevid(video, vidpath, msg)
        
        # Writing Audio
        audpath = vidpath.strip('avi')+'wav'
        wavfile.write(audpath, sample_rate, audio)
        
        # Combine
        newvidpath = vidpath.strip('.avi')+'_combined.avi'
        command = ("ffmpeg -i %s -i %s -c:v copy -c:a aac %s -hide_banner -loglevel error" % (vidpath, audpath, newvidpath))
        output = subprocess.call(command, shell=True, stdout=None)
            
    def test(self, opt, dataset):
        if os.path.exists(opt.attack_dir):
            rmtree(opt.attack_dir)
        os.makedirs(opt.attack_dir)
        
        # What to perturb
        if not opt.perturb_audio and not opt.perturb_video:
            raise ValueError("Target Modality not specified")
        if opt.perturb_audio:
            raise Exception("Audio attack - Not Supported Yet")
        # Gradient Descent
        if opt.descent:
            opt.epsilon = -opt.epsilon
        # Update per iter
        opt.a = opt.epsilon / float(opt.itersteps)
        
        # Accuracy counter
        correct = 0
        done = 0
        all_del_dist = []
        
        # Loop over all examples in test set
        for i in range(0, len(dataset)):
            try:
                # Send the data and label to the device
                videofile = dataset[i]
                
                # For Saving
                reference = os.path.basename(videofile).strip('.avi')
                opt.reference = reference

                if not videofile.endswith('avi'):
                    videofile = os.path.join(videofile, os.path.basename(videofile)+'.avi')

                # Preprocess
                imtv, cct, min_length = self.preprocess(opt, videofile)
                sample_rate, org_audio = wavfile.read(os.path.join(opt.tmp_dir,opt.reference,'audio.wav'))
                
                # Set requires_grad attribute of tensor. Important for Attack
                imtv = Variable(imtv.data.cuda(), requires_grad=True)
                cct = Variable(cct.data.cuda(), requires_grad=True)
                org_imtv = imtv
                org_cct = cct

                # Forward pass the data through the model
                im_feat, cc_feat, tS = self.gen_feat(opt, imtv, cct, min_length)
                # Only Video as this point
                imtv_grads = []
                imtv_diffs = []
                
                # Initial Offset
                print("Before Adversarial Attack")
                offset, dist, conf = self.get_offset(opt, im_feat, cc_feat, tS)

                for i in tqdm(range(opt.itersteps)):
                    # Forward pass the data through the model
                    im_feat, cc_feat, tS = self.gen_feat(opt, imtv, cct, min_length)
                    
                    # Distance
                    mdist = torch.mean(self.get_mdist(opt, im_feat, cc_feat))

                    # Maximise dist - Since we do gradient ascent, we MAXIMISE the loss defined below (for positive epsilon)
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
                        perturbed_cct = fgsm_attack(cct, opt.a, cct_grad)
                    else:
                        perturbed_cct = cct
                    if opt.perturb_video:
                        perturbed_imtv = fgsm_attack(imtv, opt.a, imtv_grad)
                    else:
                        perturbed_imtv = imtv
                    
                    # Update
                    imtv_grads.append(imtv.grad.cpu().detach())
                    imtv_diffs.append((perturbed_imtv-imtv).cpu().detach())
                    imtv = Variable(perturbed_imtv.data.cuda(), requires_grad=True)
                    cct = Variable(perturbed_cct.data.cuda(), requires_grad=True)

                # Re-classify the perturbed image
                im_feat, cc_feat, tS = self.gen_feat(opt, perturbed_imtv, perturbed_cct, min_length)
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
                    # Special case for saving 0 epsilon examples
                    if (opt.epsilon == 0):
                        print("Fooled")
                else:
                    # Save some adv examples for visualization later
                    print("Fooled")
                
                # Saving Videos
                f = "{}/{}/Adv_eps_{}.avi".format(opt.attack_dir, os.path.basename(videofile).strip('.avi'), opt.epsilon)
                audio_1, video_1 = self.postprocess(opt, perturbed_imtv, perturbed_cct)
                msg="Offset={},Dist={},Conf={}".format(round(float(offset_adv), 2),round(float(dist_adv), 2),round(float(conf_adv),2))
                self.save(opt, org_audio, video_1, f, sample_rate, msg)

                f = "{}/{}/Original.avi".format(opt.attack_dir, os.path.basename(videofile).strip('.avi'))
                audio_2, video_2 = self.postprocess(opt, org_imtv, org_cct)
                msg = "Offset={}, Dist={}, Conf={}".format(round(float(offset), 2), round(float(dist), 2), round(float(conf), 2))
                self.save(opt, org_audio, video_2, f, sample_rate, msg)

                for i in range(opt.itersteps):
                    f = "{}/{}/Grad_{}.avi".format(opt.attack_dir, os.path.basename(videofile).strip('.avi'), i)
                    grad_video = self.postprocess_grad(imtv_grads[i])
                    self.savevid(grad_video, f, "")

                    f = "{}/{}/Diff_{}.avi".format(opt.attack_dir, os.path.basename(videofile).strip('.avi'), i)
                    diff_video = self.postprocess_diff(imtv_diffs[i])
                    self.savevid(diff_video, f, "")
                print("-------------------------------------------")
                
            except Exception as e:
                print(e)
                pass
        
        # Calculate final accuracy for this epsilon
        final_acc = correct/float(done)
        avg_del_dist = np.mean(all_del_dist)
        print("Epsilon: {}\tTest Accuracy with threshold {} = {} / {} = {}\nAverage increase in distance = {}".format(opt.epsilon,
                                                                                                                      opt.threshold,
                                                                                                                      correct, done,
                                                                                                                      final_acc,
                                                                                                                      avg_del_dist))

        # Return the accuracy and an adversarial example
        return final_acc
            
# FGSM attack code - One Iter step of BIM attack
def fgsm_attack(image, epsilon, data_grad, min_lvl = 0, max_lvl = 255):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    if min_lvl is not None and max_lvl is not None:
        perturbed_image = torch.clamp(perturbed_image, min_lvl, max_lvl)
    # Return the perturbed image
    return perturbed_image

def evaluate(opt):
    if opt.filename != '':
        filename = opt.filename
        print(filename)
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
    parser.add_argument('--itersteps', type=int, default=10, help='Number of iterative steps')
    parser.add_argument('--perturb_audio', help='Whether to perturb audio', action="store_true")
    parser.add_argument('--perturb_video', help='Whether to perturb video', action="store_true")
    parser.add_argument('--evaluation_mode', help='If enabled, runs evaluation on filename instead', action="store_true")
    parser.add_argument('--maxrange', type=int, default='20', help='If using Dataset, range till which to run')
    parser.add_argument('--descent', help='If enabled, runs gradient descent on loss', action="store_true")

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