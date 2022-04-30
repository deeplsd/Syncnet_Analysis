from numpy import array
import numpy as np

def ema(s, n):
    """
    returns an n period exponential moving average for
    the time series s

    s is a list ordered from oldest (index 0) to most
    recent (index -1)
    n is an integer

    returns a numeric array of the exponential
    moving average
    """
    s = array(s)
    ema = []
    j = 1

    #get n sma first and calculate the next n period ema
    sma = sum(s[:n]) / n
    multiplier = 2 / float(1 + n)
    ema.append(sma)

    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(( (s[n] - sma) * multiplier) + sma)

    #now calculate the rest of the values
    for i in s[n+1:]:
        tmp = ( (i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)

    return array(ema)

def streak(s, t = 100):
    z = np.zeros_like(s)
    counter = 0
    for i in range(len(s)):
        if s[i] == 1:
            counter = 0
        else:
            counter += 1
        if counter < t:
            z[i] = 1
    streak = np.zeros_like(s)
    i = 0
    streak_indices = []
    while i < len(s):
        if z[i] == 1:
            j = i
            while z[j] == 1:
                j += 1
            if j - i > 10*t:
                streak_indices.append([i, j])
                for k in range(i, j):
                    streak[k] = 1
            i = j
        else:
            i += 1
    merged_SI = [streak_indices[0]]
    ms = 0
    for s in range(1, len(streak_indices)):
        if streak_indices[s][0] < merged_SI[ms][1] + 20:
            merged_SI[ms][1] = streak_indices[s][1]
        else:
            merged_SI.append(streak_indices[s])
            ms += 1
    return streak, np.array(merged_SI)