#!/usr/local/bin/python3
# Note, if any output has NAN in it, we drop the entire episode from the calculation.

import glob
import re
import numpy as np

def string_to_float(input_list):
    listed = []
    for element in input_list.split(','):
        listed.append(float(element))

    listed = np.array(listed)

    return listed


files = glob.glob('measurements*.txt')
output = open('output.txt', 'a')
#f2 = open('ue_2_sinr.txt', 'a')
#f3 = open('ue_1_tx_power.txt', 'a')
#f4 = open('ue_2_tx_power.txt', 'a')

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

    # Cleanup...
    sinr1 = re.sub(pattern, '', sinr1)
    sinr2 = re.sub(pattern, '', sinr2)
    txpower1 = re.sub(pattern, '', txpower1)
    txpower2 = re.sub(pattern, '', txpower2)

    sinr1 = string_to_float(sinr1) 
    sinr2 = string_to_float(sinr2) 
    txpower1 = string_to_float(txpower1) 
    txpower2 = string_to_float(txpower2) 

episodes = []
pattern = '_([0-9]+)_'
for filename in files:
	episode = re.findall(pattern, filename)
	episodes.append(episode[0])


episodes = np.array(episodes).astype(int)
print('Convergence episode: {}'.format(np.percentile(episodes, 60)))
output.write('Convergence episode: {}'.format(np.percentile(episodes, 60)))

sinr_average = 10*np.log10(10 ** (np.mean([sinr1, sinr2])/10.))

sinr1_average = 10*np.log10(10 ** (np.mean(sinr1)/10.))
sinr2_average = 10*np.log10(10 ** (np.mean(sinr2)/10.))

txpower1_average = np.mean(txpower1)
txpower2_average = np.mean(txpower2)
txpower_average = np.mean([txpower1, txpower2])

print('Mean SINR for UE 1: {} dB'.format(sinr1_average))
print('Mean SINR for UE 2: {} dB'.format(sinr2_average))
print('Mean BS 1 TX Power: {} W'.format(txpower1_average))
print('Mean BS 2 TX Power: {} W'.format(txpower2_average))
print('--')
print('Mean SINR: {} dB'.format(sinr_average))
print('Mean BS TX Power: {} W'.format(txpower_average))

output.write('Mean SINR for UE 1: {} dB'.format(sinr1_average))
output.write('Mean SINR for UE 2: {} dB'.format(sinr2_average))
output.write('Mean BS 1 TX Power: {} W'.format(txpower1_average))
output.write('Mean BS 2 TX Power: {} W'.format(txpower2_average))
output.write('Mean SINR: {} dB'.format(sinr_average))
output.write('Mean BS TX Power: {} W'.format(txpower_average))

output.close()
