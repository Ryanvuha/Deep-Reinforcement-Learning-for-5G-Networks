#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:45:37 2019

@author: farismismar
"""

import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.ticker import MultipleLocator, FuncFormatter

os.chdir('/Users/farismismar/Desktop/deep')
MAX_EPISODES_DEEP = 1000
    
def plot_primary(X,Y, title, xlabel, ylabel, filename='plot.pdf'):
    fig = plt.figure(figsize=(8,5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 16
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    plt.title(title)
    plt.xlabel(xlabel)
    
    ax = fig.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    # Format the ticklabel to be 2 raised to the power of `x`
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(2**x)))
    
    ax.set_autoscaley_on(False)
    
    plot_, = ax.plot(X, Y, 'k^-') #, label='ROC')

#    ax.set_xlim(xmin=0.15, xmax=0.55)
    
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, MAX_EPISODES_DEEP)
    
    plt.grid(True)
#    plt.legend([plot_], ['ROC'], loc='upper right')
    fig.tight_layout()
    plt.savefig('figures/{}'.format(filename), format='pdf')
    plt.show()

def plot_primary_two(X, Y1, Y2, title, xlabel, ylabel, filename='plot.pdf'):
    fig = plt.figure(figsize=(8,5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 16
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    plt.title(title)
    plt.xlabel(xlabel)
    
    ax = fig.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    # Format the ticklabel to be 2 raised to the power of `x`
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(2**x)))
    
    ax.set_autoscaley_on(False)
    
    plot1_, = ax.plot(X, Y1, 'k^-')
    plot2_, = ax.plot(X, Y2, 'bo-')

    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(max(Y1, Y2)))
    
    plt.grid(True)
    plt.legend([plot1_, plot2_], ['User 1', 'User 2'], loc='lower left')
    fig.tight_layout()
    plt.savefig('figures/{}'.format(filename), format='pdf')
    plt.show()
    
    
def plot_primary_three(X1, Y1, X2, Y2, title, xlabel, ylabel, filename='plot.pdf'):
    fig = plt.figure(figsize=(8,5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 16
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    plt.title(title)
    plt.xlabel(xlabel)
    
    ax = fig.gca()
    
    ax.set_autoscaley_on(False)
    
    plot1_, = ax.plot(X1, Y1, 'k^-')
    plot2_, = ax.plot(X2, Y2, 'bo-')

    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(max(Y1, Y2)))
    
    plt.grid(True)
    plt.legend([plot1_, plot2_], ['User 1', 'User 2'], loc='lower left')
    fig.tight_layout()
    plt.savefig('figures/{}'.format(filename), format='pdf')
    plt.show()
    
def plot_actions(actions, max_timesteps_per_episode, filename='plot.pdf'):
    env_actions = 8
    
    # TODO convert actions
    action_convert = actions
    
    fig = plt.figure(figsize=(8,5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 16
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    plt.xlabel('Transmit Time Interval (1 ms)')
    plt.ylabel('Action')
    
     # Only integers                                
    ax = fig.gca()
    ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%0g'))
    ax.xaxis.set_ticks(np.arange(0, max_timesteps_per_episode))
    
    ax.set_xlim(xmin=0, xmax=max_timesteps_per_episode - 1)
    
    ax.set_autoscaley_on(False)
    ax.set_ylim(-1, env_actions)
    
    plt.grid(True)
        
    plt.step(np.arange(len(actions)), action_convert_tx, color='b', label='Power Control -- BS 1')
    plt.step(np.arange(len(actions)), action_convert_int, color='b', label='Power Control -- BS 2')
    plt.savefig('figures/{}'.format(filename), format="pdf")
    plt.show(block=True)
    plt.close(fig)
    return
    
X = np.log2(np.array([4,8,16,64,128]))

##############################
# 1) Convergence vs Antenna Size
Y = [136,257,320,756,801]
plot_primary(X,Y, r'\textbf{Convergence vs. Antenna Size}', r'$M_\text{ULA}$', r'Convergence Episode', 'convergence.pdf')
##############################

##############################
# 2) Average achievable rate (or SNR) vs number of antennas
sinr1_4 = 10*np.log10(np.mean(10**(np.array([10.835746599489875, 26.318898050758378, 49.78828568247876, 25.414634594091822, 35.86099477442613, 49.043715019745534])/10.)))
sinr2_4 = 10*np.log10(np.mean(10**(np.array([16.511039960636154, 48.89391657981651, 38.616537476478484, 30.119719141289703, 14.297965471319818, 49.49941008441611])/10.)))

sinr1_8 = 10*np.log10(np.mean(10**(np.array([38.405754290002314, 11.663539094122887, 15.336834968865286, 22.824449629645294, 23.09081614281933, 27.86879421609101])/10.)))
sinr2_8 = 10*np.log10(np.mean(10**(np.array([6.820947301959588, 5.070450562845724, 27.310014009702964, 31.4852825967887, 51.09229687101583, 25.88712096141056])/10.)))

sinr1_16 = 10*np.log10(np.mean(10**(np.array([27.574224141509717, 34.424253954767096, 27.84703777006621, 24.861552592596926, 14.668541629790948, 19.834010446505644])/10.)))
sinr2_16 = 10*np.log10(np.mean(10**(np.array([27.588854787948822, 12.330098058860347, 19.51459793043395, 24.99922111445268, 9.231984132981095, 5.81334676100275])/10.)))

sinr1_64 = 10*np.log10(np.mean(10**(np.array([40.19761863285069, 6.403479714874388, 14.572068693493696, 45.566053964378945, 22.5904692854604, 33.61120878372856])/10.)))
sinr2_64 = 10*np.log10(np.mean(10**(np.array([13.443008443601014, 31.505682094174567, 26.48205680922512, 13.888175358024366, 20.869003754324453, 30.01461802410934])/10.)))

sinr1_128 = 10*np.log10(np.mean(10**(np.array([22.11667160412126, 22.281258855396562, 27.224955766786152, 4.81083495755204, 26.907376780107914, 51.77742491393877])/10.)))
sinr2_128 = 10*np.log10(np.mean(10**(np.array( [14.627372540617834, 37.53032535218256, 19.566259262829163, 27.880438744722493, 8.0358277072062, 4.972794659179764])/10.)))

Y1 = [sinr1_4, sinr1_8, sinr1_16, sinr1_64, sinr1_128]
Y2 = [sinr2_4, sinr2_8, sinr2_16, sinr2_64, sinr2_128]

plot_primary_two(X,Y1,Y2, r'\textbf{Achievable SNR vs. Antenna Size}', r'$M_\text{ULA}$', r'Average Achievable SNR [dB]', 'achievable_snr.pdf')
##############################

##############################
# 3) Coverage CDF
# TODO
##############################


##############################
# 4) SNR vs. transmit power
tx_power1_4 = 10*np.log10(np.mean(10**(np.array([44.93632713703306, 41.93632713703306, 42.93632713703306, 41.93632713703306, 41.93632713703306, 42.93632713703306])/10.)))
tx_power2_4 = 10*np.log10(np.mean(10**(np.array([39.71254462117593, 39.71254462117593, 39.71254462117593, 39.71254462117593, 38.71254462117593, 38.71254462117593])/10.)))

tx_power1_8 = 10*np.log10(np.mean(10**(np.array([41.66492387048368, 38.66492387048368, 39.66492387048368, 39.66492387048368, 38.66492387048368, 38.66492387048368])/10.)))
tx_power2_8 = 10*np.log10(np.mean(10**(np.array([43.10951224225554, 43.10951224225554, 43.10951224225554, 42.10951224225553, 42.10951224225553, 43.10951224225554])/10.)))

tx_power1_16 = 10*np.log10(np.mean(10**(np.array([42.57082022076274, 43.570820220762734, 44.57082022076274, 44.57082022076274, 43.570820220762734, 43.570820220762734])/10.)))
tx_power2_16 = 10*np.log10(np.mean(10**(np.array([37.00583694424782, 37.00583694424782, 37.00583694424782, 37.00583694424782, 37.00583694424782, 34.00583694424781])/10.)))

tx_power1_64 = 10*np.log10(np.mean(10**(np.array([39.846938617202696, 39.846938617202696, 40.846938617202696, 41.846938617202696, 41.846938617202696, 41.846938617202696])/10.)))
tx_power2_64 = 10*np.log10(np.mean(10**(np.array([42.42999200352201, 45.429992003522, 45.429992003522, 45.429992003522, 42.42999200352201, 45.429992003522])/10.)))

tx_power1_128 = 10*np.log10(np.mean(10**(np.array([42.66603067788266, 42.66603067788266, 42.66603067788266, 41.66603067788266, 41.66603067788266, 42.66603067788266])/10.)))
tx_power2_128 = 10*np.log10(np.mean(10**(np.array([38.04424717730333, 35.044247177303326, 34.044247177303326, 34.044247177303326, 34.044247177303326, 34.044247177303326])/10.)))

X1 = [tx_power1_4, tx_power1_8, tx_power1_16, tx_power1_64, tx_power1_128]
X2 = [tx_power2_4, tx_power2_8, tx_power2_16, tx_power2_64, tx_power2_128]


plot_primary_three(X1,Y1,X2,Y2, r'\textbf{Achievable SNR vs. Transmit Power}', r'Average Transmit Power [W]', r'Average Achievable SNR [dB]', 'snr_vs_tx_power.pdf')
##############################

##############################
# 5) SNR vs. transmit power
