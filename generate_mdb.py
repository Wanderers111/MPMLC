from api.generate_conformation_encode import generate_and_encode
import pickle
import pandas as pd
from multiprocessing.pool import Pool
import time

time1 = time.time()
table = pd.read_csv('./zinc15_druglike_clean_canonical_max60.smi', header=None)
smis = table[0].values.tolist()
processing = Pool(10, maxtasksperchild=1000)
zlibs_list = []
for smi_i in smis:
    zlibs_list.append(processing.apply_async(generate_and_encode, args=(smi_i,)))
processing.close()
processing.join()
zlibs_final_list = [zlib_i.get() for zlib_i in zlibs_list]
with open('dataset.mdb', 'wb') as f:
    pickle.dump(zlibs_final_list, f)
print(time.time()-time1)
