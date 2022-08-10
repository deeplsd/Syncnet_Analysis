import os
import json
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Phoneme types
plosives = ['p', 'b', 't', 'd', 'k', 'g']
fricatives = ['f', 'v', 's', 'z', 'S', 'Z', 'C', 'j', 'x', 'h']
sonorants = ['m', 'n', 'N', 'l', 'r']

# Plotting Utility Function
def plotdict(X, freq_below_th, ylab, xlab, img_file = 'PhoneImps.jpg'):
    odstore_dict = OrderedDict(sorted(X.items(), key=lambda x:x[1], reverse=True))
    k = list(odstore_dict.keys())[:-10]
    k = [ki.strip('<').strip('>') for ki in k]
    v = list(odstore_dict.values())[:-10]
    fig = plt.figure(figsize = (10, 2.3))
    c = ['lightgray' if ki in freq_below_th else 'blue' for ki in k]
    plt.bar(k, v, color = c, align='center')
    plt.ylabel(ylab)
    m = np.mean(list(X.values()))
    plt.axhline(y=m, color='y', linestyle='--')
    plt.text(33.1, m+0.00005, 'Mean Phoneme Importance', fontsize=10, va='center', ha='center')
    plt.tick_params(axis='y', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
    plt.savefig(img_file)
    print("Phoneme Importance bar graph written to {}".format(img_file))
    
def main():
    # Lab file inside each directory
    phonemes = 'phonemes.lab'
    timings = '/raid/nayak/Merkel/corpus/timings.txt'
    dir_ = '/raid/nayak/Merkel/corpus'
    segments = 'abs_segments.json'

    # Open Timings and high-gradient time-intervals
    with open(segments) as json_file:
        segs = json.load(json_file)
    with open(timings) as f:
        snips = f.readlines()

    # Collecting snippets
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

    # Collect duration of overlap of each phoneme with high-gradient regions
    print("INFO: Computing duration of overlap of each phoneme with high-gradient regions")
    store_dict = {}
    store_arr = []
    for k, v in tqdm(all_snips.items(), total = len(all_snips)):
        snip = v
        try:
            seg = segs[k]
        except:
            continue
        snip_date = snip['snip_file'].split('/')[-2].split('_')[0]
        t1 = snip['time_start']
        t2 = snip['time_end']
        lab_file = os.path.join(dir_, snip_date, phonemes)
        with open('/raid/nayak/Merkel/corpus/{}/phonemes.lab'.format(snip_date)) as f:
            phoneme_timings = f.readlines()
        for sg in seg:
            gt1 = sg[0]+t1
            gt2 = sg[1]+t1
            for pht in phoneme_timings:
                t1, t2, ph = pht.strip().split('\t')
                t1 = float(t1)
                t2 = float(t2)
                if t1 < gt2 and gt1 < t2:
                    store_arr.append((ph, t1, t2, gt1, gt2))
                    if ph in store_dict.keys():
                        store_dict[ph] += min(gt2, t2) - max(gt1, t1)
                    else:
                        store_dict[ph] = min(gt2, t2) - max(gt1, t1)

    # Collect total duration of each phoneme
    print("INFO: Computing total duration of each phoneme")
    all_store_dict = {}
    for k, v in tqdm(all_snips.items(), total = len(all_snips)):
        snip = v
        try:
            seg = segs[k]
        except:
            continue
        snip_date = snip['snip_file'].split('/')[-2].split('_')[0]
        t1 = snip['time_start']
        t2 = snip['time_end']
        lab_file = os.path.join(dir_, snip_date, phonemes)
        with open('/raid/nayak/Merkel/corpus/{}/phonemes.lab'.format(snip_date)) as f:
            phoneme_timings = f.readlines()
        for pht in phoneme_timings:
            t1, t2, ph = pht.strip().split('\t')
            t1 = float(t1)
            t2 = float(t2)
            if ph in all_store_dict.keys():
                all_store_dict[ph] += t2 - t1
            else:
                all_store_dict[ph] = t2 - t1

    # Compute relative importance as (high-gradient overlap duration / total duration)
    X = {}
    for k, v in store_dict.items():
        X[k] = v / all_store_dict[k]

    # Marking phonemes having total duration of less than a threshold value
    _max = max(all_store_dict.values())
    freq_below_th = [k for k, v in all_store_dict.items() if v < _max/50]

    # Plot bar-chart with relative importances
    plotdict(X, freq_below_th, "Rel. Phoneme Importance", "Phoneme (de)")
    
if __name__ == '__main__':
    main()