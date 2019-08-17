#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 2019  This is for the 5 point data.

@author: farismismar
"""

import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.ticker import MultipleLocator, FuncFormatter

import pandas as pd
import matplotlib2tikz

os.chdir('/Users/farismismar/Desktop/summaries')

MAX_EPISODES_DEEP = 1600
MIN_EPISODES_DEEP = 750

def sum_rate(sinr):
    output = []
    for s in sinr:
        s_linear = 10 ** (s / 10.)
        c = np.log2(1 + s_linear)
        output.append(c)

    return output
    

def generate_sum_rate(data1, data2, data3):
    c1 = np.mean(sum_rate(data1))
    c2 = np.mean(sum_rate(data2))
    c3 = np.mean(sum_rate(data3))

    fig = plt.figure(figsize=(10.24,7.68))
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    X = np.log2(np.array([4,8,16,32,64]))
    
    ax = fig.gca()
    
    ax.xaxis.set_major_locator(MultipleLocator(1))
    # Format the ticklabel to be 2 raised to the power of `x`
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(2**x)))


    plot_, = ax.plot(X, [c1, c2, c3], 'ko-')

#    ax.set_xlim(xmin=0.15, xmax=0.55)
    ax.set_xlabel(r'$M_\text{ULA}$')
    ax.set_ylabel(r'$C$ [bps/Hz]')
    ax.set_ylim(4, 12)
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sumrate.pdf', format="pdf")
    matplotlib2tikz.save('figures/sumrate.tikz')
    plt.close(fig)    

def plot_ccdf(T):
    fig = plt.figure(figsize=(10.24, 7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']   
    
    labels = T.columns

    num_bins =  50
    i = 0
    for data in T:
        data_ = T[data].dropna()

        counts, bin_edges = np.histogram(data_, bins=num_bins, density=True)
        ccdf = 1 - np.cumsum(counts) / counts.sum()
        lw = 1 + i
        ax = fig.gca()
        style = '-'
        ax.plot(bin_edges[1:], ccdf, style, linewidth=lw)

    labels = [r'$M_\text{ULA} = 4$', r'$M_\text{ULA} = 8$', r'$M_\text{ULA} = 16$', r'$M_\text{ULA} = 32$', r'$M_\text{ULA} = 64$']    

    plt.grid(True)
    plt.tight_layout()
    ax.set_xlabel('$\gamma$')
    ax.set_ylabel('$1 - F_\Gamma(\gamma)$')        
    ax.legend(labels, loc="lower left")
    plt.savefig('ccdf.pdf', format='pdf')
    matplotlib2tikz.save('figures/ccdf.tikz')
    plt.close(fig)    
    
def generate_ccdf(data1, data2, data3, data4, data5):
    fig = plt.figure(figsize=(10.24,7.68))
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    for data in [data1, data2, data3, data4, data5]:
        # sort the data:
        data_sorted = np.sort(data)
        
        # calculate the proportional values of samples
        p = 1. * np.arange(len(data)) / (len(data) - 1)
        
        ax = fig.gca()
    
        # plot the sorted data:
        ax.plot(data_sorted, 1 - p)
    

    labels = [r'$M_\text{ULA} = 4$', r'$M_\text{ULA} = 8$', r'$M_\text{ULA} = 16$', r'$M_\text{ULA} = 32$', r'$M_\text{ULA} = 64$']
    ax.set_xlabel('$\gamma$')
    ax.set_ylabel('$1 - F_\Gamma(\gamma)$')        
    ax.legend(labels, loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ccdf.pdf', format="pdf")
    matplotlib2tikz.save('figures/ccdf.tikz')
    plt.close(fig)    

def plot_primary(X,Y, title, xlabel, ylabel, ymin=0, ymax=MAX_EPISODES_DEEP, filename='plot.pdf'):
    fig = plt.figure(figsize=(10.24,7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    #plt.title(title)
    plt.xlabel(xlabel)
    
    ax = fig.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    # Format the ticklabel to be 2 raised to the power of `x`
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(2**x)))
    
    ax.set_autoscaley_on(False)
    
    plot_, = ax.plot(X, Y, 'k^-') #, label='ROC')

#    ax.set_xlim(xmin=0.15, xmax=0.55)
    
    ax.set_ylabel(ylabel)
    ax.set_ylim(ymin, ymax)
    
    plt.grid(True)
#    plt.legend([plot_], ['ROC'], loc='upper right')
    fig.tight_layout()
    plt.savefig('{}'.format(filename), format='pdf')
    matplotlib2tikz.save('figures/{}.tikz'.format(filename))

           
def plot_secondary(X,Y1, Y2, Y3, Y4, title, xlabel, y1label, y2label, y1max, y2max, filename='plot.pdf'):
    fig = plt.figure(figsize=(10.24,7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    #plt.title(title)
    plt.xlabel(xlabel)
    plt.grid(True, axis='both', which='both')
        
    ax = fig.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    # Format the ticklabel to be 2 raised to the power of `x`
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(2**x)))
    
    ax.set_autoscaley_on(False)
    ax_sec = ax.twinx()
    
    plot1_, = ax.plot(X, Y1, 'k^-')
    plot2_, = ax.plot(X, Y2, 'bo--')
    plot3_, = ax_sec.plot(X, Y3, 'r^-')
    plot4_, = ax_sec.plot(X, Y4, 'go--')

#    ax.set_xlim(xmin=0.15, xmax=0.55)
    
    ax.set_ylabel(y1label)
    ax_sec.set_ylabel(y2label)
    
    ax.set_ylim(0, y1max)
    ax_sec.set_ylim(0, y2max)
    
    plt.grid(True)
    plt.legend([plot1_, plot2_, plot3_, plot4_], ['TX Power JB-PCIC', 'TX Power Optimal', 'SINR JB-PCIC', 'SINR Optimal'], loc='lower left')
    fig.tight_layout()
    plt.savefig('{}'.format(filename), format='pdf')
    matplotlib2tikz.save('figures/{}.tikz'.format(filename))
    
def plot_primary_two(X, Y1, Y2, title, xlabel, ylabel, filename='plot.pdf'):
    fig = plt.figure(figsize=(10.24,7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    #plt.title(title)
    plt.xlabel(xlabel)
    
    ax = fig.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    # Format the ticklabel to be 2 raised to the power of `x`
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(2**x)))
    
    ax.set_autoscaley_on(False)
    
    plot1_, = ax.plot(X, Y1, 'k^-')
    plot2_, = ax.plot(X, Y2, 'ro--')

    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(max(Y1), max(Y2))*1.02)
    
    plt.grid(True)
    plt.legend([plot1_, plot2_], ['JB-PCIC', 'Optimal'], loc='lower left')
    fig.tight_layout()
    plt.savefig('{}'.format(filename), format='pdf')
    matplotlib2tikz.save('figures/{}.tikz'.format(filename))
    
    
def plot_primary_three(X1, Y1, X2, Y2, title, xlabel, ylabel, filename='plot.pdf'):
    fig = plt.figure(figsize=(10.24,7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    #plt.title(title)
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
    matplotlib2tikz.save('figures/{}}.tikz'.format(filename))
    plt.savefig('{}'.format(filename), format='pdf')
    
#def plot_actions(actions, max_timesteps_per_episode, filename='plot.pdf'):
#    env_actions = 6
#    
#    # TODO convert actions
#    action_convert = actions
#    
#    fig = plt.figure(figsize=(10.24,7.68))
#    plt.rc('text', usetex=True)
#    plt.rc('font', family='serif')
#    matplotlib.rcParams['text.usetex'] = True
#    matplotlib.rcParams['font.size'] = 20
#    matplotlib.rcParams['text.latex.preamble'] = [
#        r'\usepackage{amsmath}',
#        r'\usepackage{amssymb}']
#    
#    plt.xlabel('Transmit Time Interval (1 ms)')
#    plt.ylabel('Action')
#    
#     # Only integers                                
#    ax = fig.gca()
#    ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%0g'))
#    ax.xaxis.set_ticks(np.arange(0, max_timesteps_per_episode))
#    
#    ax.set_xlim(xmin=0, xmax=max_timesteps_per_episode - 1)
#    
#    ax.set_autoscaley_on(False)
#    ax.set_ylim(-1, env_actions)
#    
#    plt.grid(True)
#        
#    plt.step(np.arange(len(actions)), action_convert_tx, color='b', label='Power Control -- BS 1')
#    plt.step(np.arange(len(actions)), action_convert_int, color='b', label='Power Control -- BS 2')
#    plt.savefig('{}'.format(filename), format="pdf")
#    plt.show(block=True)
#    plt.close(fig)
#    return

##############################
# 1) Convergence vs Antenna Size
# Obtained from the percentile of convergence.
X = np.log2(np.array([4,8,16,32,64]))
Y = [1006.06, 1031.5 , 1097.6, 1131.46, 1378.060]
plot_primary(X,Y, r'\textbf{Convergence vs. Antenna Size}', r'$M$', r'Convergence Episode ($\zeta$)', MIN_EPISODES_DEEP, MAX_EPISODES_DEEP, 'convergence.pdf')

##############################
# 2) Coverage CDF

def compute_distributions(optimal=False, cutoff=0):
    df_final = pd.DataFrame()
    
    for M in [4, 8, 16, 32, 64]:
        if optimal:
            df_1 = pd.read_csv('M={} opt/ue_1_sinr.txt'.format(M), sep=',', header=None, index_col=False).transpose().dropna()
            df_2 = pd.read_csv('M={} opt/ue_2_sinr.txt'.format(M), sep=',', header=None, index_col=False).transpose().dropna()
        else:
            df_1 = pd.read_csv('M={}/ue_1_sinr.txt'.format(M), sep=',', header=None, index_col=False).transpose().dropna()
            df_2 = pd.read_csv('M={}/ue_2_sinr.txt'.format(M), sep=',', header=None, index_col=False).transpose().dropna()
        
        df = pd.concat([df_1, df_2], axis=0, ignore_index=True)
        df = df.astype(float)
        sinr = np.array(df)
        indexes = (sinr >= cutoff)
        df_sinr = pd.DataFrame(sinr[indexes])
        df_sinr.columns = ['sinr_{}'.format(M)]
        
        if optimal:
            df_3 = pd.read_csv('M={} opt/ue_1_tx_power.txt'.format(M), sep=',', header=None, index_col=False).transpose().dropna()
            df_4 = pd.read_csv('M={} opt/ue_2_tx_power.txt'.format(M), sep=',', header=None, index_col=False).transpose().dropna()
        else:
            df_3 = pd.read_csv('M={}/ue_1_tx_power.txt'.format(M), sep=',', header=None, index_col=False).transpose().dropna()
            df_4 = pd.read_csv('M={}/ue_2_tx_power.txt'.format(M), sep=',', header=None, index_col=False).transpose().dropna()
        
        df = pd.concat([df_3, df_4], axis=0, ignore_index=True)
        df = df.astype(float)
        
        tx_power = np.array(df)
        df_power = pd.DataFrame(tx_power[indexes])
        df_power.columns = ['tx_power_{}'.format(M)]
    
        df_final = pd.concat([df_final, df_sinr, df_power], axis=1)
        
    return df_final

df_final_ = compute_distributions()
df_final = df_final_.values
df_final = df_final.T
sinr_4, tx_power_4, sinr_8, tx_power_8, sinr_16, tx_power_16, sinr_32, tx_power_32, sinr_64, tx_power_64 = df_final 
generate_ccdf(sinr_4, sinr_8, sinr_16, sinr_32, sinr_64)
plot_ccdf(df_final_[['sinr_4', 'sinr_8', 'sinr_16', 'sinr_32', 'sinr_64']])

##############################
# 3) SINR and transmit power vs M
# TODO, instead of np.mean, try np.nanpercentile.

df_final_opt_ = compute_distributions(optimal=True)
df_final_opt = df_final_opt_.values
df_final_opt = df_final_opt.T
sinr_4_optimal, tx_power_4_optimal, sinr_8_optimal, tx_power_8_optimal, sinr_16_optimal, tx_power_16_optimal, sinr_32_optimal, tx_power_32_optimal, sinr_64_optimal, tx_power_64_optimal = df_final_opt 

q = 50
tx_power_4_agg = 10*np.log10(10 ** (np.nanpercentile(tx_power_4, q)/10.))
tx_power_8_agg = 10*np.log10(10 ** (np.nanpercentile(tx_power_8, q)/10.))
tx_power_16_agg = 10*np.log10(10 ** (np.nanpercentile(tx_power_16, q)/10.))
tx_power_32_agg = 10*np.log10(10 ** (np.nanpercentile(tx_power_32, q)/10.))
tx_power_64_agg = 10*np.log10(10 ** (np.nanpercentile(tx_power_64, q)/10.))

sinr_avg_4 = 10*np.log10(10 ** (np.nanpercentile(sinr_4, q)/10.))
sinr_avg_8 = 10*np.log10(10 ** (np.nanpercentile(sinr_8, q)/10.))
sinr_avg_16 = 10*np.log10(10 ** (np.nanpercentile(sinr_16, q)/10.))
sinr_avg_32 = 10*np.log10(10 ** (np.nanpercentile(sinr_32, q)/10.))
sinr_avg_64 = 10*np.log10(10 ** (np.nanpercentile(sinr_64, q)/10.))

##################################################################################################################

tx_power_4_optimal_agg = 10*np.log10(10 ** (np.nanpercentile(tx_power_4_optimal, q)/10.))
tx_power_8_optimal_agg = 10*np.log10(10 ** (np.nanpercentile(tx_power_8_optimal, q)/10.))
tx_power_16_optimal_agg = 10*np.log10(10 ** (np.nanpercentile(tx_power_16_optimal, q)/10.))
tx_power_32_optimal_agg = 10*np.log10(10 ** (np.nanpercentile(tx_power_32_optimal, q)/10.))
tx_power_64_optimal_agg = 10*np.log10(10 ** (np.nanpercentile(tx_power_64_optimal, q)/10.))

sinr_avg_4_optimal = 10*np.log10(10 ** (np.nanpercentile(sinr_4_optimal, q)/10.))
sinr_avg_8_optimal = 10*np.log10(10 ** (np.nanpercentile(sinr_8_optimal, q)/10.))
sinr_avg_16_optimal = 10*np.log10(10 ** (np.nanpercentile(sinr_16_optimal, q)/10.))
sinr_avg_32_optimal = 10*np.log10(10 ** (np.nanpercentile(sinr_32_optimal, q)/10.))
sinr_avg_64_optimal = 10*np.log10(10 ** (np.nanpercentile(sinr_64_optimal, q)/10.))

Y1 = [tx_power_4_agg, tx_power_8_agg, tx_power_16_agg, tx_power_32_agg, tx_power_64_agg]
Y2 = [sinr_avg_4, sinr_avg_8, sinr_avg_16, sinr_avg_32, sinr_avg_64]

Y3 = [tx_power_4_optimal_agg, tx_power_8_optimal_agg, tx_power_16_optimal_agg, tx_power_32_optimal_agg, tx_power_64_optimal_agg] # power optimal
Y4 = [sinr_avg_4_optimal, sinr_avg_8_optimal, sinr_avg_16_optimal, sinr_avg_32_optimal, sinr_avg_64_optimal] # sinr optimal

plot_secondary(X,Y1,Y3,Y2,Y4, r'\textbf{Median TX Power and SINR vs. Antenna Size}', r'$M$', r'Median Transmit Power [dBm]', r'Median Achievable SINR [dB]', 46, 67, 'achievable_sinr_power.pdf')

##############################
# 4) Average achievable rate (or SNR) vs number of antennas
#plot_primary(X,Y2, r'\textbf{Achievable SINR vs. Antenna Size}', r'$M_\text{ULA}$', r'Average Achievable SNR [dB]', 50, 'achievable_snr.pdf')
plot_primary_two(X,sum_rate(Y2),sum_rate(Y4), r'\textbf{Median Sum-Rate vs. Antenna Size}', r'$M$', r'$C$ [bps/Hz]', 'sumrate.pdf')
##############################

