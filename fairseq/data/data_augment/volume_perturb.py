import random
import pdb

def volume_perturb(wavform,low=-1.6,high=1.6):
    #pdb.set_trace()
    volume_factor = 10 ** (random.uniform(low, high) / 20)
    wavform = wavform.float()
    wavform *= volume_factor
    return wavform
