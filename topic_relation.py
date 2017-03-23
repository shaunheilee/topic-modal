#coding=utf8
# ========================================================
#   Copyright (C) 2017 All rights reserved.
#   
#   filename : topic_relation.py
#   author   : ***
#   date     : 2016-05-12
#   desc     : 
# ========================================================
import os
import sys
import numpy as np
import random
import gibbs

tels = {}
itels = {}
words = {}
iwords = {}
on_dt = []
off_tw = []

c = 0
fp = open(sys.argv[1])
while True:
    line = fp.readline()
    if not line:
        break
    items = line.strip().split('\t')
    tels.setdefault(items[0], c)
    itels.setdefault(c, items[0])
    c += 1
    on_dt.append(map(int, items[1:]))
fp.close()

c = 0
fp = open(sys.argv[2])
while True:
    line = fp.readline()
    if not line:
        break
    items = line.strip().split('\t')
    words.setdefault(items[0], c)
    iwords.setdefault(c, items[0])
    c += 1
    off_tw.append(map(int, items[1:]))
fp.close()

t2t = []
for i in range(100):
    t2t.append([0] * 50)

tokens = []
fp = open(sys.argv[3])
while True:
    line = fp.readline()
    if not line:
        break
    items = line.strip().split("\t")
    t1 = -1
    t2 = -1
    if items[0] in tels:
        t1 = tels[items[0]]
    else:
        continue
    if items[1] in words:
        t2 = words[items[1]]
    else:
        continue
    off_t = int(items[2]) - 1
    on_t = int(100 * random.random())
    t2t[on_t][off_t] += 1
    tokens.append((t1, t2, on_t, off_t))
fp.close()

print >> sys.stderr, "tokens load done! tokens: %d" % len(tokens)


on_dt = np.array(on_dt)
off_tw = np.array(off_tw)
t2t = np.array(t2t)
tokens = np.array(tokens)

gibbs.gibbs_sample(tokens, on_dt, t2t, len(tokens), 50, 100)
