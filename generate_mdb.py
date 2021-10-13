from api.generate_conformation_encode import generate_and_encode
import pickle
import pandas as pd
from multiprocessing.pool import Pool
import time

time1 = time.time()
table = pd.read_table('./ChEMBL.txt', header=None)
smis = table[0].values.tolist()

t = 0
smis_len = len(smis)
for i in range(smis_len, 100000):
    processing = Pool(30, maxtasksperchild=1000)
    zlibs_list = []
    for smi_i in smis[i: min(i+100000, smis_len)]:
        zlibs_list.append(processing.apply_async(generate_and_encode, args=(smi_i,)))
    processing.close()
    processing.join()
    zlibs_final_list = [zlib_i.get() for zlib_i in zlibs_list]
    with open(f'dataset_{t}.mdb', 'wb') as f:
        pickle.dump(zlibs_final_list, f)
    t += 1
print(time.time()-time1)
