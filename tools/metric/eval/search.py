import time
import os
import sys
import numpy as np
import faiss
import pickle



def clac_mAP(resultfile, gtfile, topk = 100):
    gt = {}
    input = open(gtfile, "r").readlines()
    for i, item in enumerate(input):
        tts = item.strip().split()
        Q = tts[0]
        if len(tts) > 1:
            gt[Q] = list(set(tts[1:]))
        else:
            gt[Q] = []

    failcount = 0
    okcount = 0
    cnt_r = 0
    mAP = 0
    for line in open(resultfile, "rb"):
        info = line.strip().split()
        query, rs = info[0], info[1:]
        Q = query
        if Q not in gt:
            failcount += 1
            continue
        okcount += 1
        R = gt[Q]
        if len(R) == 0:
            continue
        bound = min(len(rs), topk)
        ind = np.zeros(bound)
        for n, ntem in enumerate(rs[:bound]):
            if ntem in R:
                ind[n] = 1.0
            else:
                ind[n] = 0.0
        AP = 0
        num = min(len(R), topk)
        for k in range(bound):
            if ind[k] == 1:
                right = np.sum(ind[:k+1])
                precision = right * 1.0 / (k+1)
                AP += precision / num
        mAP += AP
        cnt_r += 1
    mAP = mAP / cnt_r
    print("mAP:", "%.2f%%"%(mAP * 100))



def search_gpu(query_path, refer_path, output, topk=100):
    queryfeas, queryconts = loadFeaFromPickle(query_path)
    referfeas, referconts = loadFeaFromPickle(refer_path)
    assert(queryfeas.shape[1] == referfeas.shape[1])
    dim = int(queryfeas.shape[1])
    print("=> query feature shape: {}".format(queryfeas.shape), file=sys.stderr)
    print("=> refer feature shape: {}".format(referfeas.shape), file=sys.stderr)
    
    start = time.time()
    ngpus = faiss.get_num_gpus()
    print("=> search use gpu number of GPUs: {}".format(ngpus), file=sys.stderr)
    cpu_index = faiss.IndexFlat(dim, faiss.METRIC_INNER_PRODUCT)   # build the index
    gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
            cpu_index
            )
    gpu_index.add(referfeas)                  # add vectors to the index print(index.ntotal)
    print("=> building gpu index success, \
           total index number: {}".format(gpu_index), file=sys.stderr)
    distance, ind = gpu_index.search(queryfeas, int(topk))
    assert(distance.shape == ind.shape)
    end = time.time()
    print("=> searching total use time {}".format(end - start), file=sys.stderr)
    outdic = {}
    for key_id in range(queryfeas.shape[0]):
        querycont = queryconts[key_id]
        searchresult = [(referconts[ind[key_id][i]], distance[key_id][i]) \
                         for i in range(len(distance[key_id]))]
        outdic[querycont] = searchresult
    print("=> convert search gpu result to output format success")
    pickle.dump(outdic, open(output,"wb"), protocol=2)


def loadFeaFromPickle(feafile):
    feadic = pickle.load(open(feafile,'rb'))
    fea_items = feadic.items()
    names = [fea[0] for fea in fea_items]
    feas = [fea[1].reshape(-1) for fea in fea_items]
    feas = np.array(feas)
    return feas, names



if __name__ == "__main__":
    if len(sys.argv)>1 :
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('tools.py command', file=sys.stderr)