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

os.chdir('/Users/farismismar/Desktop/test')
MAX_EPISODES_DEEP = 5000

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

def plot_primary(X,Y, title, xlabel, ylabel, ymax=MAX_EPISODES_DEEP, filename='plot.pdf'):
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
    ax.set_ylim(0, ymax)
    
    plt.grid(True)
#    plt.legend([plot_], ['ROC'], loc='upper right')
    fig.tight_layout()
    plt.savefig('figures/{}'.format(filename), format='pdf')
    plt.show()

def plot_secondary(X,Y1, Y2, title, xlabel, y1label, y2label, y1max, y2max, filename='plot.pdf'):
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
    ax_sec = ax.twinx()
    
    plot1_, = ax.plot(X, Y1, 'k^-')
    plot2_, = ax.plot(X, Y2, 'bo-')

#    ax.set_xlim(xmin=0.15, xmax=0.55)
    
    ax.set_ylabel(y1label)
    ax_sec.set_ylabel(y2label)
    
    ax.set_ylim(0, y1max)
    ax_sec.set_ylim(0, y2max)
    
    plt.grid(True)
    plt.legend([plot1_, plot2_], ['TX Power', 'SINR'], loc='upper right')
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

##############################
# 1) Convergence vs Antenna Size

X = np.log2(np.array([4,16,64]))
Y = [190,1522,4952]
plot_primary(X,Y, r'\textbf{Convergence vs. Antenna Size}', r'$M_\text{ULA}$', r'Convergence Episode', MAX_EPISODES_DEEP, 'convergence.pdf')

##############################
# 2) Coverage CDF

sinr_progress_4=[45.406333165900264, 39.496271622713635, 0.17947180297845577, 25.919786722896685, 6.2468642970844535, 34.70415331481286, 25.169030230541903, 38.03421480160987, 45.98958945461571, 49.63233990561758, 17.619093044883435, 34.96546107709224, 29.17321654737636, 25.33941087281679, 13.176597456488555]
sinr_ue2_progress_4=[9.436742752740926, 16.592896876256468, 0.44367526677877694, 28.842233062129115, 42.729073352792255, 24.95094355192642, 0.2185505513622467, 4.826206091191754, 27.456409819555404, 11.84869807153648, 17.73602903713485, 35.07877710523694, 30.472731226911428, -2.687043189512215, 38.56753865690956]

sinr_progress_16= [42.03827264285556, 32.145235233855175, 35.790152683215325, 26.37042936011901, 60.647983884584775, 2.3416771144193635, 39.31985685353039, 19.056136067590394, 56.473384055806044, 45.55728101172983, 2.329404874683204, 40.26341852244872, 0.20867895795885977, 45.70090604911872, 33.28426837679486]
sinr_ue2_progress_16= [4.752822474424621, 5.662334363920375, 47.869169306744425, 31.52089491966674, 35.99469716166971, 22.511106752296225, 49.016691680291586, 39.551112361585496, 30.435289431440296, 4.948972139388453, 9.41977588890213, 7.004964964371392, 15.62743511544742, 29.437728350770005, 24.86637749682163]

sinr_progress_64= [36.23921581842194, 26.883122703168652, 23.835562782048886, 15.817045575950633, 11.375818929426496, 1.0337552007170854, 14.67838131743232, 7.174028101861209, 20.70486470645062, 16.52023981537622, 3.4684998693167164, 20.906871366234373, 2.9954744003122307, 33.781612128752954, 15.364582412361496]
sinr_ue2_progress_64= [55.842344254529046, 45.45925157932935, 38.04301144341745, 25.71972925587904, 29.398258688447427, 56.36274548552558, 42.75453394001307, 80.61063730439763, 21.016308941944533, 76.52439034213951, 14.538493365014807, 94.09048910939853, 46.867427699825804, 57.39551622459578, 23.482560234011277]

sinr1_4 = 10*np.log10(np.mean(10**(np.array(sinr_progress_4)/10.)))
sinr2_4 = 10*np.log10(np.mean(10**(np.array(sinr_ue2_progress_4)/10.)))

sinr1_16 = 10*np.log10(np.mean(10**(np.array(sinr_progress_16)/10.)))
sinr2_16 = 10*np.log10(np.mean(10**(np.array(sinr_ue2_progress_16)/10.)))

sinr1_64 = 10*np.log10(np.mean(10**(np.array(sinr_progress_64)/10.)))
sinr2_64 = 10*np.log10(np.mean(10**(np.array(sinr_ue2_progress_64)/10.)))

sinr_4 = sinr_progress_4 + sinr_ue2_progress_4
sinr_16 = sinr_progress_16 + sinr_ue2_progress_16
sinr_64 = sinr_progress_64 + sinr_ue2_progress_64

generate_ccdf(sinr_4, sinr_16, sinr_64)

##############################
# 3) SNR vs. transmit power

tx_power1_progress_4 =[42.55760458767428, 42.55760458767428, 42.55760458767428, 42.55760458767428, 41.55760458767428, 41.55760458767428, 41.55760458767428, 41.55760458767428, 41.55760458767428, 41.55760458767428, 41.55760458767428, 41.55760458767428, 41.55760458767428, 41.55760458767428, 41.55760458767428]
tx_power2_progress_4 =[41.298709527147025, 41.298709527147025, 41.298709527147025, 42.298709527147025, 42.298709527147025, 42.298709527147025, 43.298709527147025, 43.298709527147025, 43.298709527147025, 43.298709527147025, 43.298709527147025, 42.298709527147025, 42.298709527147025, 42.298709527147025, 42.298709527147025]

tx_power1_progress_16 = [39.32747346343462, 39.32747346343462, 39.32747346343462, 39.32747346343462, 39.32747346343462, 40.32747346343462, 40.32747346343462, 41.327473463434615, 41.327473463434615, 42.32747346343462, 42.32747346343462, 42.32747346343462, 43.327473463434615, 43.327473463434615, 43.327473463434615]
tx_power2_progress_16 = [41.12655094063604, 41.12655094063604, 42.12655094063604, 42.12655094063604, 42.12655094063604, 42.12655094063604, 42.12655094063604, 42.12655094063604, 41.12655094063604, 41.12655094063604, 42.12655094063604, 43.12655094063604, 43.12655094063604, 44.12655094063604, 43.12655094063604]

tx_power1_progress_64 = [42.3255768011861, 41.325576801186095, 41.325576801186095, 40.3255768011861, 39.3255768011861, 39.3255768011861, 38.3255768011861, 37.3255768011861, 36.3255768011861, 35.3255768011861, 34.3255768011861, 33.3255768011861, 32.3255768011861, 31.325576801186102, 31.325576801186102]
tx_power2_progress_64 = [40.52907803270644, 40.52907803270644, 40.52907803270644, 40.52907803270644, 40.52907803270644, 40.52907803270644, 40.52907803270644, 40.52907803270644, 40.52907803270644, 40.52907803270644, 40.52907803270644, 40.52907803270644, 40.52907803270644, 40.52907803270644, 39.529078032706444]

tx_power_4 = np.mean(10**(np.array([tx_power1_progress_4,tx_power2_progress_4])/10.)) / 1000.
tx_power_16 = np.mean(10**(np.array([tx_power1_progress_16,tx_power2_progress_16])/10.)) / 1000.
tx_power_64 = np.mean(10**(np.array([tx_power1_progress_64,tx_power2_progress_64])/10.)) / 1000.

sinr_avg_4 = 10*np.log10(10 ** (np.mean(sinr_4)/10.))
sinr_avg_16 = 10*np.log10(10 ** (np.mean(sinr_16)/10.))
sinr_avg_64 = 10*np.log10(10 ** (np.mean(sinr_64)/10.))

Y1 = [tx_power_4, tx_power_16, tx_power_64]
Y2 = [sinr_avg_4, sinr_avg_16, sinr_avg_64]

plot_secondary(X,Y1,Y2, r'\textbf{Average TX Power and SINR vs. Antenna Size}', r'$M_\text{ULA}$', r'Average Transmit Power [dBm]', r'Average Achievable SINR [dB]', 46, 50, 'achievable_sinr_power.pdf')

##############################
# 4) Average achievable rate (or SNR) vs number of antennas
plot_primary(X,Y2, r'\textbf{Achievable SINR vs. Antenna Size}', r'$M_\text{ULA}$', r'Average Achievable SNR [dB]', 50, 'achievable_snr.pdf')
##############################
