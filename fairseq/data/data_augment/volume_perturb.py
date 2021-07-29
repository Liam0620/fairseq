import random
import pdb

def volume_perturb(wavform):
    #pdb.set_trace()
    volume_factor = 10 ** (random.uniform(-1.6, 1.6) / 20)
    wavform = wavform.float()
    wavform *= volume_factor
    return wavform
