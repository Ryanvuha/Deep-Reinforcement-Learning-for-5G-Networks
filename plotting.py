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
MAX_EPISODES_DEEP = 3500

def generate_ccdf(data1, data2, data3):
    fig = plt.figure(figsize=(8,5))
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 16
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    for data in [data1, data2, data3]:
        # sort the data:
        data_sorted = np.sort(data)
        
        # calculate the proportional values of samples
        p = 1. * np.arange(len(data)) / (len(data) - 1)
        
        ax = fig.gca()
    
        # plot the sorted data:
        ax.plot(data_sorted, 1 - p)
    

    labels = [r'$M_\text{ULA} = 4$', r'$M_\text{ULA} = 16$', r'$M_\text{ULA} = 64$']
    ax.set_xlabel('$\gamma$')
    ax.set_ylabel('$1 - F_\Gamma(\gamma)$')        
    ax.legend(labels, loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/ccdf.pdf', format="pdf")
    plt.show()
    plt.close(fig)    

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
    ax.set_ylim(0, max(max(Y1), max(Y2)))
    
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
    
    order = np.argsort(X1)
    X1 = np.array(X1)[order]
    Y1 = np.array(Y1)[order]
    
    order = np.argsort(X2)
    X2 = np.array(X2)[order]
    Y2 = np.array(Y2)[order]
    
    ax = fig.gca()
    
    ax.set_autoscaley_on(False)
    
    plot1_, = ax.plot(X1, Y1, 'k^-')
    plot2_, = ax.plot(X2, Y2, 'bo-')

    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(max(Y1), max(Y2)))
    
    plt.grid(True)
    plt.legend([plot1_, plot2_], ['User 1', 'User 2'], loc='lower left')
    fig.tight_layout()
    plt.savefig('figures/{}'.format(filename), format='pdf')
    plt.show()
    
def plot_actions(actions, max_timesteps_per_episode, filename='plot.pdf'):
    env_actions = 6
    
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

X = np.log2(np.array([4,16,64]))

##############################
# 1) Convergence vs Antenna Size
Y = [3394,2757,1182]


plot_primary(X,Y, r'\textbf{Convergence vs. Antenna Size}', r'$M_\text{ULA}$', r'Convergence Episode', 'convergence.pdf')
##############################

##############################
# 2) Average achievable rate (or SNR) vs number of antennas

sinr_progress_4=[15.269691024577272, 20.311089476991427, 48.615024979020085, 47.34922389158125, 10.897027298145392, 27.26697419938023, 5.020379123600519, 35.667165360818295, 51.92061298782593, 4.852028964397882]
sinr_ue2_progress_4= [12.360831589883787, 15.665119510418608, 7.380971444694939, 21.45515361797103, 34.800868702704776, 36.91964675108764, 9.000293958938023, 12.371869133225427, 21.544041817682945, 43.36658454445025]

#sinr_progress_8= np.zeros(40)
#sinr_ue2_progress_8=  np.zeros(40)

sinr_progress_16=  [53.077851080487406, 34.42255256709865, 22.7102926330047, 25.210163023593974, 51.56736353622368, 31.645304497224274, 34.617272144325064, 28.758796002337785, 3.882438962270507, 52.49551516088063]
sinr_ue2_progress_16=  [19.446623091409535, 11.935211633595642, 13.997952128792459, 67.12608215500781, 38.109136818735344, 22.748091633302586, 1.1948730712546531, 3.1993661045111432, 20.296657930224754, 8.479904585289102]


sinr_progress_64= [62.21950622911579, 54.91984677801131, 14.255905237562104, 32.97015492447518, 51.832604018426274, 6.421871617168798, 46.05296080334688, 4.322422339207473, 30.280230727727385, 39.59472515457065]
sinr_ue2_progress_64=[31.60430040715973, 12.535114083354838, 17.60788189241724, 85.38569051358535, 25.259264850325657, 10.038290721569512, 72.44991195685968, 34.39169974824226, 44.221370831520574, 33.00308662086222]


sinr1_4 = 10*np.log10(np.mean(10**(np.array(sinr_progress_4)/10.)))
sinr2_4 = 10*np.log10(np.mean(10**(np.array(sinr_ue2_progress_4)/10.)))

#sinr1_8 = 10*np.log10(np.mean(10**(np.array(sinr_progress_8)/10.)))
#sinr2_8 = 10*np.log10(np.mean(10**(np.array(sinr_ue2_progress_8)/10.)))

sinr1_16 = 10*np.log10(np.mean(10**(np.array(sinr_progress_16)/10.)))
sinr2_16 = 10*np.log10(np.mean(10**(np.array(sinr_ue2_progress_16)/10.)))

sinr1_64 = 10*np.log10(np.mean(10**(np.array(sinr_progress_64)/10.)))
sinr2_64 = 10*np.log10(np.mean(10**(np.array(sinr_ue2_progress_64)/10.)))

#sinr1_128 = 10*np.log10(np.mean(10**(np.array(sinr_progress_128)/10.)))
#sinr2_128 = 10*np.log10(np.mean(10**(np.array(sinr_ue2_progress_128)/10.)))

X = np.log2(np.array([4,16,64]))
Y1 = [sinr1_4, sinr1_16, sinr1_64]
Y2 = [sinr2_16, sinr2_16,  sinr2_64]

plot_primary_two(X,Y1,Y2, r'\textbf{Achievable SNR vs. Antenna Size}', r'$M_\text{ULA}$', r'Average Achievable SNR [dB]', 'achievable_snr.pdf')
##############################

##############################
# 3) Coverage CDF
##############################
sinr_4 = sinr_progress_4 + sinr_ue2_progress_4
#sinr_8 = sinr_progress_8 + sinr_ue2_progress_8
sinr_16 = sinr_progress_16 + sinr_ue2_progress_16
sinr_64 = sinr_progress_64 + sinr_ue2_progress_64
#sinr_128 = sinr_progress_128 + sinr_ue2_progress_128

generate_ccdf(sinr_4, sinr_16, sinr_64)

##############################
# 4) SNR vs. transmit power

tx_power1_progress_4 = [31.83232576121119, 30.83232576121119, 31.83232576121119, 32.832325761211195, 33.83232576121119, 33.83232576121119, 34.83232576121119, 35.83232576121119, 35.83232576121119, 36.832325761211195]
tx_power2_progress_4 = [43.94844331228302, 43.94844331228302, 43.94844331228302, 43.94844331228302, 43.94844331228302, 43.94844331228302, 43.94844331228302, 43.94844331228302, 43.94844331228302, 43.94844331228302]

tx_power1_progress_16 = [37.148170356368205, 37.148170356368205, 37.148170356368205, 36.1481703563682, 36.1481703563682, 36.1481703563682, 36.1481703563682, 36.1481703563682, 36.1481703563682, 36.1481703563682]
tx_power2_progress_16 = [39.277819714451525, 38.277819714451525, 38.277819714451525, 38.277819714451525, 38.277819714451525, 38.277819714451525, 39.277819714451525, 38.277819714451525, 38.277819714451525, 38.277819714451525]

tx_power1_progress_64 = [31.83232576121119, 30.83232576121119, 31.83232576121119, 32.832325761211195, 33.83232576121119, 33.83232576121119, 34.83232576121119, 35.83232576121119, 35.83232576121119, 36.832325761211195]
tx_power2_progress_64 = [43.94844331228302, 43.94844331228302, 43.94844331228302, 43.94844331228302, 43.94844331228302, 43.94844331228302, 43.94844331228302, 43.94844331228302, 43.94844331228302, 43.94844331228302]

tx_power1_4 = np.mean(10**(np.array(tx_power1_progress_4)/10.)) / 1000.
tx_power2_4 = np.mean(10**(np.array(tx_power2_progress_4)/10.)) / 1000.

#tx_power1_8 = 10*np.log10(np.mean(10**(np.array(tx_power1_progress_8)/10.)))
#tx_power2_8 = 10*np.log10(np.mean(10**(np.array(tx_power2_progress_8)/10.)))

tx_power1_16 = np.mean(10**(np.array(tx_power1_progress_16)/10.)) / 1000.
tx_power2_16 = np.mean(10**(np.array(tx_power2_progress_16)/10.)) / 1000.

tx_power1_64 = np.mean(10**(np.array(tx_power1_progress_64)/10.)) / 1000.
tx_power2_64 = np.mean(10**(np.array(tx_power2_progress_64)/10.)) / 1000.

#tx_power1_128 = 10*np.log10(np.mean(10**(np.array(tx_power1_progress_128)/10.)))
#tx_power2_128 = 10*np.log10(np.mean(10**(np.array(tx_power2_progress_128)/10.)))

X1 = [tx_power1_4, tx_power1_16, tx_power1_64]
X2 = [tx_power2_4, tx_power2_16,  tx_power2_64]


plot_primary_three(X1,Y1,X2,Y2, r'\textbf{Achievable SNR vs. Transmit Power}', r'Average Transmit Power [W]', r'Average Achievable SNR [dB]', 'snr_vs_tx_power.pdf')
##############################

##############################
# 5) SNR vs. transmit power
