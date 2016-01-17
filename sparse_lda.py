#coding=utf8
# =======================================================================
# Author: zhanzhizheng
# Created Time : Fri 16 Jan 2016 11:12:03 PM CST
#
# File Name: sparse_lda.py
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
    dtz = [] 
    twz = []

    for i in range(D):
        doc_topic.append([0]*K)
        dtz.append(set())

    for i in range(K):
        topic_word.append([0]*V)

    for i in range(V):
        twz.append(set())

    for i in range(T):
        t = tokens[i]
        did = t[0]
        wid = t[1]
        k = t[2]
        doc_topic[did][k] += 1
        topic_word[k][wid] += 1
        topic[k] += 1
        dtz[did].add(k)
        twz[wid].add(k)

    save(doc_topic, topic_word, docs, words, 0)
    for i in range(niter):
        print >> sys.stderr, 'Iteration %d\t...\t' % i,
        gibbs_sample(doc_topic, topic_word, topic, tokens, D, V, T, K, a, b, dtz, twz)
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


def gibbs_sample(dt, tw, tp, tokens, D, V, T, K, a, b, dtz, twz):
    ab = a * b
    Vb = V * b
    cc = 0

    # cache for constant
    e = 0
    for i in range(K):
        e += ( ab / (Vb + tp[i]) )
    
    f = 0
    pre_did = -1
    for ti, t in enumerate(tokens):
        did = t[0]
        wid = t[1]
        k = t[2]

        # doc_topic , topic_word and topic should not count
        # current word's topic in current document
        
        # cache f
        if pre_did != did:
            f = 0
            for idx in dtz[did]:
                f += ( dt[did][idx] * b / (Vb + tp[idx]) )
        pre_did = did

        dt[did][k] -= 1
        tw[k][wid] -= 1
        tp[k] -= 1

        if dt[did][k] == 0:
            dtz[did].remove(k)

        if tw[k][wid] == 0:
            twz[wid].remove(k)
        
        # ajusting e, f
        _tmp = (Vb + tp[k]) * (Vb + tp[k] + 1)
        #e = e + ab / (Vb + tp[k]) - ab / (Vb + tp[k] + 1)
        e = e + ab / _tmp
        #f = f + b * (dt[did][k] / (Vb + tp[k]) - (dt[did][k] + 1) / (Vb + tp[k] + 1))
        f = f + b * (dt[did][k] - Vb - tp[k]) / _tmp

        # cal G
        G = []
        _G = []

        for i, idx in enumerate(twz[wid]):
            G.append(tw[idx][wid] * (a + dt[did][idx]) / (Vb+tp[idx]) )
            _G.append(idx)
            if i > 0:
                G[i] += G[i - 1]

        if len(G) == 0:
            G.append(0)

        # sample a new topic for current word in current document
        u = random.random() * (e + f + G[-1])
        _k = -1

        if u < e:
            # smooth bucket
            _s = 0
            for i, _p in enumerate(tp):
                _s += ab / (Vb + _p)
                if u < _s:
                    _k = i
                    break
        elif u < (e + f):
            # doc-topic bucket
            u = u - e
            _s = 0
            for i, idx in enumerate(dtz[did]):
                _s += dt[did][idx] * b / ( Vb + tp[idx])
                if u < _s and _k == -1:
                    _k = idx
                    break
        else:
            # topic-word bucket
            u = u - e - f
            for i, _p in enumerate(G):
                if u < _p:
                    _k = _G[i]
                    break
        
        if _k == -1:
            print >> sys.stderr, 'ERROR!'
            sys.exit(-1)
        
        dt[did][_k] += 1
        tw[_k][wid] += 1
        tp[_k] += 1
        
        if _k not in dtz[did]:
            dtz[did].add(_k)

        twz[wid].add(_k)
        # change to new topic for the word
        tokens[ti][2] = _k
        # ajusting e,f
        _tmp = (Vb + tp[_k]) * (Vb + tp[_k] - 1)
        #e = e + ab / (Vb + tp[_k]) - ab / (Vb + tp[_k] - 1)
        e = e - ab / _tmp 
        #f = f + b * (dt[did][_k] / (Vb + tp[_k]) - (dt[did][_k] - 1) / (Vb + tp[_k] - 1))
        f = f + b * (Vb + tp[_k] - dt[did][_k]) / _tmp

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
