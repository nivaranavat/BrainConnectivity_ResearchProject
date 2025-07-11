"""
phase_utils.py
Domain-specific utilities for phase scrambling in brain network analysis.
"""
import numpy as np
import pandas as pd
import math
from source.phase_and_correlation_scramble import *
from source.utils.io_utils import readMRIFile, load_txt
from source.utils.matrix_utils import *

def phaseScramble1(data):
    """
    Phase scrambling: add random phase to each frequency component.
    """
    
    fs = np.fft.fft(data)
    pow_fs = np.abs(fs) ** 2
    phase_fs = np.angle(fs)
    phase_fsr = phase_fs.copy()
    
    #adding a random value between 0 and 2 Pi
    for i in range(len(phase_fsr)):
        add = np.random.uniform(0,2*math.pi)
        phase_fsr[i] += add
        
    #np.random.shuffle(phase_fsr)
    fsrp = np.sqrt(pow_fs) * (np.cos(phase_fsr) + 1j * np.sin(phase_fsr))
    tsr = np.fft.ifft(fsrp)
    
    #sqrt of that sum of squared of real + imag
    # sqrt of real^2+ imag^2
    
    #tsr = np.abs(tsr)
    
    return tsr

def phaseScramble2(nparray):
    """
    Phase scrambling: split phases, shuffle, and mirror.
    """
    series = pd.Series(nparray)
    fourier = np.fft.fft(series)
    pow_fs = np.abs(fourier) ** 2.
    phase_fs = np.angle(fourier)
    phase_fsr = phase_fs.copy()
    if len(nparray) % 2 == 0:
        phase_fsr_lh = phase_fsr[1:len(phase_fsr)//2]
    else:
        phase_fsr_lh = phase_fsr[1:len(phase_fsr)//2 + 1]
    np.random.shuffle(phase_fsr_lh)
    if len(nparray) % 2 == 0:
        phase_fsr_rh = -phase_fsr_lh[::-1]
        phase_fsr = np.concatenate((np.array((phase_fsr[0],)), phase_fsr_lh,
                                    np.array((phase_fsr[len(phase_fsr)//2],)),
                                    phase_fsr_rh))
    else:
        phase_fsr_rh = -phase_fsr_lh[::-1]
        phase_fsr = np.concatenate((np.array((phase_fsr[0],)), phase_fsr_lh, phase_fsr_rh))
    fsrp = np.sqrt(pow_fs) * (np.cos(phase_fsr) + 1j * np.sin(phase_fsr))
    tsr = np.fft.ifft(fsrp)
    return tsr 