from scipy.special import gamma as gammafnc
from scipy.special import gammainc, gammaincc

# Originally created for the simulation of GIG processes and adaptive truncation methods.

def incgammau(s, x):
    return gammaincc(s,x)*gammafnc(s)

def incgammal(s, x):
    return gammainc(s,x)*gammafnc(s)

