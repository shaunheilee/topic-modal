#coding=utf8
# =======================================================================
# Author: zhanzhizheng
# Created Time : Fri 15 Jan 2016 11:12:03 PM CST
#
# File Name: lda.py
# Description:
#
# =======================================================================

import os, sys
import random
import numpy as np

def load_tokens(fn, K):
    docs = {}
    idocs = {}
    words = {}
    iwords = {}
    d_idx = 0
    w_idx = 0

    tokens = []
    fp = open(fn)
    while True:
        line = fp.readline()
        if not line:
            break
        
        items = line.strip().split('\t')
        if items[0] not in docs:
            docs.setdefault(items[0], d_idx)
            idocs.setdefault(d_idx, items[0])
            d_idx +=1
        if items[1] not in words:
            words.setdefault(items[1], w_idx)
            iwords.setdefault(w_idx, items[1])
            w_idx +=1

        did = docs[items[0]]
        wid = words[items[1]]

        tokens.append([did, wid, int(random.random() * K)])


    fp.close()
    
    return idocs, iwords, tokens

def est(docs, words, tokens, K, a, b, niter):
    doc_topic = []
    topic_word = []
    topic = [0]*K
    D = len(docs)
    V = len(words)
    T = len(tokens)

    for i in range(D):
        doc_topic.append([0]*K)

    for i in range(K):
        topic_word.append([0]*V)

    for i in range(T):
        t = tokens[i]
        did = t[0]
        wid = t[1]
        k = t[2]
        doc_topic[did][k] += 1
        topic_word[k][wid] += 1
        topic[k] += 1

    save(doc_topic, topic_word, docs, words, 0)
    for i in range(niter):
        print >> sys.stderr, 'Iteration %d\t...\t' % i,
        gibbs_sample(doc_topic, topic_word, topic, tokens, D, V, T, K, a, b)
        if i % 20 == 0:
            save(doc_topic, topic_word, docs, words, i)
        print >> sys.stderr, 'Done.'

    save(doc_topic, topic_word, docs, words, niter)

def save(dt, tw, docs, words, i):
    fp1 = open('doc_topic_%d' % i, 'w')
    fp2 = open('word_topic_%d' % i, 'w')

    for did, p in enumerate(dt):
        dn = docs[did]
        print >> fp1, '%s\t%s' % (dn, '\t'.join(['%d' % x for x in p]))

    tw = np.array(tw).T.tolist()
    for wid, p in enumerate(tw):
        wn = words[wid]
        print >> fp2, '%s\t%s' % (wn, '\t'.join(['%d' % x for x in p]))

    fp1.close()
    fp2.close()


def gibbs_sample(dt, tw, tp, tokens, D, V, T, K, a, b):
    for ti, t in enumerate(tokens):
        did = t[0]
        wid = t[1]
        k = t[2]

        # doc_topic , topic_word and topic should not count
        # current word's topic in current document
        dt[did][k] -= 1
        tw[k][wid] -= 1
        tp[k] -= 1

        # cal P(z | ...)

        p = []
        
        for i in range(K):
            p.append( (a + dt[did][i]) * (b + tw[i][wid]) * 1.0 / (V * b + tp[i]) )
            if i > 0:
                p[i] += p[i - 1]
        
        # sample a new topic for current word in current document
        u = random.random() * p[-1]
        _k = -1
        for i, _p in enumerate(p):
            if u < _p:
                _k = i
                break
        
        dt[did][_k] += 1
        tw[_k][wid] += 1
        tp[_k] += 1

        # change to new topic for the word
        tokens[ti][2] = _k

def train():
    fn = sys.argv[1]
    K = int(sys.argv[2])
    a = float(sys.argv[3])
    b = float(sys.argv[4])
    niter = int(sys.argv[5])

    print >>sys.stderr, 'Loading Tokens...'
    docs, words, tokens = load_tokens(fn, K)

    print >>sys.stderr, 'Token loading Done. docs:%d words:%d' % (len(docs), len(words))

    est(docs, words, tokens, K, a, b, niter)

    print >>sys.stderr, 'Done!'

train()
