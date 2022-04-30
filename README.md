# Adversarial attack on SyncNet

https://user-images.githubusercontent.com/45385843/156050801-2dc0ddd9-d7be-4cf9-b726-def3f1b17f5f.mp4

https://user-images.githubusercontent.com/45385843/156051375-85b8aac6-a714-4395-875b-10534d8c5b09.mp4

https://user-images.githubusercontent.com/45385843/156051526-01d9541a-7912-4536-89f7-f580d1f28f30.mp4

```
usage: FGSM_attack_syncnet.py [-h] [--model MODEL] [--batch_size BATCH_SIZE] [--vshift VSHIFT] [--out_dir OUT_DIR] [--data_dir DATA_DIR] [--filename FILENAME] [--filename_list FILENAME_LIST]
                              [--alpha ALPHA] [--epsilon EPSILON] [--threshold THRESHOLD] [--perturb_audio] [--perturb_video] [--evaluation_mode] [--maxrange MAXRANGE] [--descent]
```
```
usage: BIM_attack_syncnet.py [-h] [--model MODEL] [--batch_size BATCH_SIZE] [--vshift VSHIFT] [--out_dir OUT_DIR] [--data_dir DATA_DIR] [--filename FILENAME] [--filename_list FILENAME_LIST]
                             [--alpha ALPHA] [--epsilon EPSILON] [--threshold THRESHOLD] [--itersteps ITERSTEPS] [--perturb_audio] [--perturb_video] [--evaluation_mode] [--maxrange MAXRANGE]
                             [--descent]
```


```
Important arguments:
  --filename FILENAME            Name of the video file
  --filename_list FILENAME_LIST  Path to file with video paths
  --epsilon EPSILON              Max perturbation magnitude (>0)
  --threshold THRESHOLD          Threshold increase in distance to classify example as fooled
  --perturb_audio                Whether to perturb audio
  --perturb_video                Whether to perturb video
  --evaluation_mode              If enabled, runs evaluation on filename instead
  --descent                      If enabled, runs gradient descent on loss
  --itersteps ITERSTEPS          Number of iterative steps (BIM attack ONLY)
```
