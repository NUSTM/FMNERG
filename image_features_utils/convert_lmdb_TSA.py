'''
@Description: 
@Author: k.chen
@Date: 2019-09-22 13:42:05
@LastEditTime: 2020-04-25 10:00:47
'''
import h5py
import os
import pdb
import numpy as np
import json
import sys
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'texts','features', 'cls_prob']
import csv
import base64
csv.field_size_limit(sys.maxsize)
import sys
import pickle
import lmdb # install lmdb by "pip install lmdb"
from tqdm import tqdm

count = 0
num_file = 1
name = '/mnt/nfs-storage-titan/data/image_feature_tsv/twitter2017_2_obj.tsv'
infiles = [name]
print(infiles)

save_path = os.path.join('/mnt/nfs-storage-titan/data/image_feature_lmdb/twitter2017_2_obj.lmdb')
env = lmdb.open(save_path, map_size=1099511627776)

id_list = []
with env.begin(write=True) as txn:
    for infile in tqdm(infiles):
        with open(infile) as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter = '\t', fieldnames = FIELDNAMES)
            for (i, item) in enumerate(reader):
                img_id = str(item['image_id']).encode()
                id_list.append(img_id)
                txn.put(img_id, pickle.dumps(item))
                print("{}/{}".format(i, reader),end='\r')
    
    txn.put('keys'.encode(), pickle.dumps(id_list))
    
    
    
    
    
    
    
 
