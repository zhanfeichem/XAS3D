import numpy as np
import os

'''
"name.txt" should not contain space at the END-line.  ####ns=len(lines)

'''
NCPU=4

with open("name.txt") as f:
    lines=f.readlines()
ns=len(lines)
# ni=np.ceil(ns/NCPU)
# n_list=np.zeros(NCPU)
# for icut in range(NCPU):
#     if icut*ni>ns:
#         break
# print(icut)
# n_list[:icut-1]=ni
# n_list[icut-1:]=ni-1

ni=np.ceil(ns/NCPU)
n_list=ni*np.ones(NCPU)
n_list[NCPU-1]=ns-(NCPU-1)*ni

for j in range(NCPU):
    begin=int(   sum( n_list[:j]  )  )
    end = int(   sum(n_list[:j+1])   )
    tmp=lines[begin:end]
    with open("name"+str(j)+".txt","w") as f:
        f.writelines(tmp)

