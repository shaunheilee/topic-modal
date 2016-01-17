#coding=utf8
# ========================================================
#   Copyright (C) 2015 All rights reserved.
#   
#   filename : top_words.py
#   author   : ***
#   date     : 2015-03-12
#   desc     : 
# ======================================================== 

import sys

f = sys.argv[1]
k = int(sys.argv[2])
of = sys.argv[3]

topic_words = [[] for i in range(k)]

inFp = open(f)

while True: 
    line = inFp.readline()
    if not line:
        break
    segs = line.strip().split("\t")
    word = segs[0]
    for i in range(k):
        t = int(segs[i + 1])
        topic_words[i].append([word, t])
inFp.close()

outFp = open(of,"w")
for i in range(k):
    t = topic_words[i]
    ts = sorted(t, lambda x, y : cmp(x[1], y[1]), reverse = True)[0:20]
    outFp.write("%d th topic : \n" %(i, ))
    for tw in ts:
        outFp.write("\t%s  : %d\n" %(tw[0],tw[1]))
outFp.close()
