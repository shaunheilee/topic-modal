import os
import sys
cimport cython
from libc.stdlib cimport malloc, free, rand, srand
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
def gibbs_sample(np.ndarray[np.int_t, ndim=2] tokens, np.ndarray[np.int_t, ndim=2] on_dt, np.ndarray[np.int_t, ndim=2] t2t, int sz, int ep, int onsz):
    cdef int * uids
    cdef int * vids
    cdef int * onids
    cdef int * offids
    cdef int uid
    cdef int vid
    cdef int onid
    cdef int offid
    cdef int s_onid
    cdef double pp
    cdef int i
    cdef int j
    cdef int ti
    cdef np.ndarray[np.double_t, ndim=1] v

    uids = <int *> malloc(sz * cython.sizeof(int))
    vids = <int *> malloc(sz * cython.sizeof(int))
    onids = <int *> malloc(sz * cython.sizeof(int))
    offids = <int *> malloc(sz * cython.sizeof(int))

    srand(123)
    for i in xrange(sz):
        uids[i] = tokens[i][0]
        vids[i] = tokens[i][1]
        onids[i] = tokens[i][2]
        offids[i] = tokens[i][3]

    for i in xrange(ep):
        print >> sys.stderr, "iteration %d" % i
        for ti in xrange(sz):
            uid = uids[ti]
            vid = vids[ti]
            onid = onids[ti]
            offid = offids[ti]
            t2t[onid, offid] -= 1

            v = on_dt[uid, :] * t2t[:, offid] * 1.0 / np.sum(t2t, 1)

            s_onid = -1
            pp = rand() % 1000 / 1000.0 * np.sum(v)
            for j in xrange(onsz):
                if j > 0:
                    v[j] += v[j - 1]
                if pp < v[j]:
                    s_onid = j
                    break

            t2t[s_onid, offid] += 1
            onids[ti] = s_onid
            
            print >> sys.stderr, "%d\r" % ti,

        fp = open("t2t_%d" % i, 'w')
        for j in xrange(onsz):
            print >> fp, "\t".join(["%d" % x for x in t2t[j, :]])
        fp.close()
    
    free(uids)
    free(vids)
    free(onids)
    free(offids)
    return 0
