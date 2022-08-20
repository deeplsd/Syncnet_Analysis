# Video Interpretability

We analyse the gradients on the video input (obtained using [Integrated Gradients](https://arxiv.org/pdf/1703.01365.pdf)), to try and observe which attribute of the face SyncNet is paying attention to while computing LSE-D.
We find that SyncNet consistently focusses near the lips of the speaker, which matches human intuition regarding audio-visual synchrony.

![icmi2face drawio](https://user-images.githubusercontent.com/45385843/185760875-59f2fa40-b982-438a-94e0-e03d1e420d54.png)

### Usage
> Need to modify according to your data-directory location and structure
```
python InteGrads.py --filename /raid/dsaha/Merkel_one/2014-06-21_0008/2014-06-21_0008.avi
```
