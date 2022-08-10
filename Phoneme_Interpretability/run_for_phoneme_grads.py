import random
import os
from grad_phons import *
from tqdm import tqdm
import json

# Set paths
timings = '/raid/nayak/Merkel/corpus/timings.txt'
dir_ = '/raid/nayak/Merkel/corpus'
with open(timings) as f:
    snips = f.readlines()

# Collect video-audio snippets (Only those corresponding to which we have phonemes.lab files)
curr_idx = 0
old_snip_id = None
all_snips = {}
for snip in snips:
    x = snip.split('|')
    if x[0].split('-')[0] >= '2013':
        continue
    if x[0] != old_snip_id:
        old_snip_id = x[0]
        curr_idx = 0
    else:
        curr_idx += 1
    snip_id = '{}_{}'.format(x[0], '%02d' % curr_idx)
    snip_details = {
        'snip_file': '/raid/dsaha/Merkel_one/{}/{}.avi'.format(snip_id, snip_id),
        'time_start': float(x[1]),
        'time_end': float(x[2]),
        'sentence': x[3]
    }
    all_snips[snip_id] = snip_details

# Initialize Syncnet
opt = initialize()

# Collect all segments
allsegs = {}
counter = 0
print("Getting all audio segments")
for k, v in tqdm(all_snips.items(), total = len(all_snips)):
    video = v['snip_file']
    tsegs = run(opt, video, None, None);
    if tsegs:
        allsegs[k] = tsegs
    counter += 1

# Write to json file
with open('abs_segments.json', 'w') as fp:
    json.dump(allsegs, fp)