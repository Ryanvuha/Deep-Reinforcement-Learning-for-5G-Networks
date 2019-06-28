#!/usr/local/bin/python3
# Note, if any output has NAN in it, we drop the entire episode from the calculation.

import glob
import re
import numpy as np

files = glob.glob('measurements*.txt')
f1 = open('ue_1_sinr.txt', 'a')
f2 = open('ue_2_sinr.txt', 'a')
f3 = open('ue_1_tx_power.txt', 'a')
f4 = open('ue_2_tx_power.txt', 'a')


pattern = re.compile('[\[\]_ \':a-z]+') # get rid of [], colons, and words.

for file in files:
    f = open(file, 'r')
    lines = f.read()
    sinr1 = lines.split(':')[1]
    sinr2 = lines.split(':')[2]
    txpower1 = lines.split(':')[3]
    txpower2 = lines.split(':')[4]
    
    if ('nan' in sinr1) or ('nan' in sinr2):
        continue
    
    # Clean up sinr1, 2 by replacing pattern with ''
    f1.write('{},'.format(re.sub(pattern, '', sinr1)))
    f2.write('{},'.format(re.sub(pattern, '', sinr2)))

    # Clean up tx1, 2 by replacing pattern with ''
    f3.write('{},'.format(re.sub(pattern, '', txpower1)))
    f4.write('{},'.format(re.sub(pattern, '', txpower2)))

f4.close()
f3.close()
f2.close()
f1.close()
f.close()


	
