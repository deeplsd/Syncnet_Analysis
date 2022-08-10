# Adversarial attacks on SyncNet
We perform the following attacks on SyncNet aimed at generating adversarial examples -   

### Fast Gradient Sign Attack (FGSM)
This attack uses the gradient of the loss w.r.t the input data, then adjusts the input data to maximize the loss. In the case of SyncNet, the attack is aimed at maximising LSE-D (averaged over multiple offset values). This algorithm adjusts **each frame of the video (or audio)** by a taking a small step in the direction that will **maximize LSE-D**.
Implementation based on [this PyTorch tutorial](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html).


```                              
usage: FGSM_attack_syncnet.py [-h] [--model MODEL] [--batch_size BATCH_SIZE] [--vshift VSHIFT] [--out_dir OUT_DIR] [--data_dir DATA_DIR] [--filename FILENAME]
                              [--filename_list FILENAME_LIST] [--alpha ALPHA] [--epsilon EPSILON] [--threshold THRESHOLD] [--perturb_audio] [--perturb_video] [--evaluation_mode]
                              [--maxrange MAXRANGE] [--descent] [--shift SHIFT] [--trim_t1 TRIM_T1] [--trim_t2 TRIM_T2]
```

```
(Important arguments)
  --filename FILENAME            Name of the video file (Required)
  --filename_list FILENAME_LIST  Path to file with video paths
  --epsilon EPSILON              Max perturbation magnitude (>0)
  --threshold THRESHOLD          Threshold increase in distance to classify example as fooled
  --perturb_audio                Whether to perturb audio (Required)
  --perturb_video                Whether to perturb video (Required)
  --evaluation_mode              If enabled, runs evaluation on filename instead
  --descent                      If enabled, runs gradient descent on loss
  --shift                        Shift video by t seconds
  --trim_t1 --trim_t2            Trim video from t1 seconds to t2 seconds
```

### Basic Iteractive Method (BIM) 
BIM improves upon FGSM by iteratively adjusting the input data by smaller step size over multiple iterations. For our case, it often leads to a higher increase in LSE-D metric compared to FGSM. 
> Note: Our current BIM implementation does not support audio attacks.

```
usage: BIM_attack_syncnet.py [-h] [--model MODEL] [--batch_size BATCH_SIZE] [--vshift VSHIFT] [--out_dir OUT_DIR] [--data_dir DATA_DIR] [--filename FILENAME] [--filename_list FILENAME_LIST]
                             [--alpha ALPHA] [--epsilon EPSILON] [--threshold THRESHOLD] [--itersteps ITERSTEPS] [--perturb_video] [--evaluation_mode] [--maxrange MAXRANGE] [--descent]
```

```
(New arguments)
  --itersteps ITERSTEPS          Number of iterative steps (EPSILON / ITERSTEPS adjustment per-iteration)
```
### Simple Black-box Adversarial Attacks (SimBA)
SimBA proposes a simple black-box adversarial attack method, where we randomly select from a set of orthogonal directions and take a step along that direction, if it increases the probability of the target class (or increases the LSE-D in our case). For the orthogonal directions, they try Cartesian basis and Discrete cosine basis. The implementation was adapted from [this GitHub repository](https://github.com/cg563/simple-blackbox-attack). In our experiments, white-box attacks were found to be much more effective at generating adversarial examples.
> Note: Our current SimBA implementation does not support DCT basis attacks on audio.
```
usage: simba_syncnet.py [-h] [--model MODEL] [--batch_size BATCH_SIZE] [--vshift VSHIFT] [--out_dir OUT_DIR] [--data_dir DATA_DIR] [--filename FILENAME]
                        [--filename_list FILENAME_LIST] [--alpha ALPHA] [--epsilon EPSILON] [--threshold THRESHOLD] [--perturb_audio] [--perturb_video] [--evaluation_mode]
                        [--maxrange MAXRANGE] [--descent] [--shift SHIFT] [--pixel_attack]
```
```
(New arguments)
  --pixel_attack        If enabled, runs blackbox attack on pixels, otherwise on DCT basis
```

## Example
```
python FGSM_attack_syncnet.py --filename /raid/dsaha/Merkel_one/2013-10-19_0015/2013-10-19_0015.avi --perturb_video --epsilon 2
```

https://user-images.githubusercontent.com/45385843/156050801-2dc0ddd9-d7be-4cf9-b726-def3f1b17f5f.mp4

https://user-images.githubusercontent.com/45385843/156051375-85b8aac6-a714-4395-875b-10534d8c5b09.mp4

https://user-images.githubusercontent.com/45385843/156051526-01d9541a-7912-4536-89f7-f580d1f28f30.mp4
