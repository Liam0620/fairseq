#!/usr/bin/env python

import io
import os
import subprocess
import linecache
import numpy as np
import soundfile as sf
import scipy.signal as ss
import random
import time
import librosa
import argparse
import textgrid
import pdb
from multiprocessing import Process


def get_line_context(file_path, line_number):
    return linecache.getline(file_path, line_number).strip()


def sfread(fname):
    y, fs = sf.read(fname)
    if fs != 16000:
        y = librosa.resample(y, fs, 16000)
    return y

def cutwav(wav, minlen, maxlen):
    if wav.shape[0] < 16000*maxlen:
        return wav
    else:
        duration = int(random.uniform(minlen,maxlen)*16000)
        start = random.randint(0, wav.shape[0]-duration)
        wav = wav[start:start+duration]
        return wav

def mixwav(fname, fnoisename, frir,snr):
    samps= sfread(fname)
    noise = cutwav(sfread(fnoisename), 5, 10)
    rir = sfread(frir)
    if len(samps.shape) > 1:
        samps = samps[:,0]
    if len(noise.shape) > 1:
        noise = noise[:,0]
    
    snr = float(snr)
    samps, samps1, samps2 = add_noise(samps, noise, rir, snr)
    return samps, samps1, samps2



def mix_snr(clean, noise, snr):
    t = np.random.normal(loc = 0.9, scale = 0.1)
    if t < 0:
        t = 1e-1
    elif t > 1:
        t = 1
    scale = t

    clean_snr = snr
    noise_snr = -snr

    clean_weight = 10**(clean_snr/20)
    noise_weight = 10**(noise_snr/20)
    for i in range(clean.shape[1]):
        clean[:, i]  = activelev(clean[:, i]) * clean_weight
        noise[:, i]  = activelev(noise[:, i]) * noise_weight
    noisy = clean + noise

    max_amp = np.zeros(clean.shape[1])
    for i in range(clean.shape[1]):
        max_amp[i] = np.max(np.abs([clean[:,i], noise[:,i], noisy[:,i]]))
        if max_amp[i] == 0:
            max_amp[i] = 1
        max_amp[i] = 1 / max_amp[i] * scale

    for i in range(noisy.shape[1]):
        noisy[:, i]= noisy[:, i] * max_amp[i]
        clean[:, i]= clean[:, i] * max_amp[i]
        noise[:, i]= noise[:, i] * max_amp[i]
    
    return noisy, clean, noise

def add_reverb(cln_wav, rir_wav):
    """
    Args:
        :@param cln_wav: the clean wav
        :@param rir_wav: the rir wav
    Return:
        :@param wav_tgt: the reverberant signal
    """
    rir_wav = np.array(rir_wav)
    rir_wav = rir_wav[:, np.newaxis]
    wav_tgt = np.zeros([cln_wav.shape[0]+(len(rir_wav)-1), rir_wav.shape[1]])
    for i in range(1):
        wav_tgt[:, i] = ss.oaconvolve(cln_wav, rir_wav[:,i]/np.max(np.abs(rir_wav[:,i])))
    return wav_tgt

def activelev(data):
    # max_val = np.max(np.abs(data))
    max_val = np.std(data)
    if max_val == 0:
        return data
    else:
        return data / max_val

def add_noise(clean, noise, rir, snr):
    random.seed(time.clock())
    if len(rir.shape)>1:
        clean = add_reverb(clean, rir[:, 0])
        noise = add_reverb(noise, rir[:, 1])
    else:
        clean = add_reverb(clean, rir)
        noise = add_reverb(noise, rir)
    clean = clean[:-(len(rir)-1)]
    noise = noise[:-(len(rir)-1)]
    snr = snr / 2


    clean_length = clean.shape[0]
    noise_length = noise.shape[0]

    if clean_length > noise_length:
        padlength = clean_length - noise_length
        padfront = random.randint(0, padlength)
        padend =  padlength - padfront
        noise = np.pad(noise, ((padfront, padend), (0, 0)),'constant', constant_values=(0,0))
        noise_selected = noise
        clean_selected = clean
        
    elif clean_length < noise_length:
        noise = noise[:clean_length]
        noise_selected = noise
        clean_selected = clean
        
    
    else:
        noise_selected = noise
        clean_selected = clean

    noisy, clean, noise = mix_snr(clean_selected, noise_selected, snr)
    return noisy, clean, noise


def run(wavlist, noiselist, rirlist, output_dir, datamode, args_text, args_mode,args_aishell4_wav_list,args_textgrid_list):


    output_text = output_dir+'/'+datamode+'/text'
    output_text = open(output_text, 'w')
    output_utt2dur = output_dir+'/'+datamode+'/utt2dur'
    output_utt2dur = open(output_utt2dur, 'w')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(output_dir+'/'+datamode+'/wav'):
        os.makedirs(output_dir+'/'+datamode+'/wav')


    for i in range(0, len(open(wavlist,'r').readlines())):
        random.seed(time.clock())
        wavidx = random.randint(0, len(open(wavlist,'r').readlines())-1)
        noiseidx = random.randint(0, len(open(noiselist,'r').readlines())-1)
        riridx = random.randint(0, len(open(rirlist,'r').readlines())-1)
        wav_path = get_line_context(wavlist, wavidx+1)
        noise_path = get_line_context(noiselist, noiseidx+1)
        rir_path = get_line_context(rirlist, riridx+1)
        random.seed(time.clock())
        snr = random.uniform(5, 20)
        outname = wav_path.split('/')[-1]
        outname = outname.split('.wav')[0]
        out, spk1, spk2 = mixwav(wav_path, noise_path, rir_path, snr)
        sf.write(output_dir+'/'+datamode+'/wav/'+outname+'_noisy.wav', out[:,0], 16000)
        text = get_line_context(args_text, i+1)
        #pdb.set_trace()
        text = text.split(' ')[1]
        output_text.write(outname+'_noisy '+text+'\n')
        output_utt2dur.write(outname+'_noisy '+str(out[:,0].shape[0]/16000)+'\n')
        output_text.flush()
        output_utt2dur.flush()


def add_noise_rir(wav_path, noise_path, rir_path):
    snr = random.uniform(5, 20)
    outname = wav_path.split('/')[-1]
    outname = outname.split('.wav')[0]
    out, spk1, spk2 = mixwav(wav_path, noise_path, rir_path, snr)
    out = out[:,0]
    return out

class NoiseRIR_Dataset():
    def __init__(self,noise_path,rir_path,low_snr,high_snr):
        super().__init__()

        self.low_snr = low_snr
        self.high_snr = high_snr
        self.noises = []
        self.rirs = []
        with open(noise_path, "r") as f_noise, open(rir_path, "r") as f_rir:
            self.root_noise = f_noise.readline().strip()
            self.root_rir = f_rir.readline().strip()
            for i, line in enumerate(f_noise):
                self.noises.append(line.strip())
            for i, line in enumerate(f_rir):
                self.rirs.append(line.strip())

    def add_noise_rir(self,wav_path):
        rand_noise_path = os.path.join(self.root_noise, random.choice(self.noises))
        rand_rir_path = os.path.join(self.root_rir, random.choice(self.rirs))
        snr = random.uniform(self.low_snr, self.high_snr)
        out, spk1, spk2 = mixwav(wav_path, rand_noise_path, rand_rir_path, snr)
        out = out[:, 0]
        return out

if __name__ == '__main__':
    num_process = 50
    parser = argparse.ArgumentParser()
    parser.add_argument("--aishell1_wav_list",
                        type=str,
                        help="aishell1_wav_list",
                        default="rawwav_list/train/aishell1_dev.txt")
    parser.add_argument("--noise_list",
                        type=str,
                        help="noise_list",
                        default="rawwav_list/train/noise_tr.txt")
    parser.add_argument("--rir_list",
                        type=str,
                        help="rir_list",
                        default="rawwav_list/train/rir_2-8s_1-5m_aishell4_tr.txt")
    parser.add_argument("--text",
                        type=str,
                        help="text",
                        default="rawwav_list/train/text_dev")
    parser.add_argument("--output_dir",
                        type=str,
                        help="output_dir for data",
                        default="data_asr")
    parser.add_argument("--mode",
                        type=str,
                        help="train or dev",
                        default="train")
    parser.add_argument("--aishell4_wav_list",
                        type=str,
                        help="aishell4_wav_list to generate training data of real-recorded aishell-4 data",
                        default="rawwav_list/train/aishell4_eval.txt")
    parser.add_argument("--textgrid_list",
                        type=str,
                        help="textgrid_list",
                        default="rawwav_list/train/aishell4_textgrid.txt")

    args = parser.parse_args()

    wavlist = args.aishell1_wav_list
    noiselist = args.noise_list
    rirlist = args.rir_list
    output_dir = args.output_dir
    datamode = args.mode
    args_text = args.text
    args_mode = args.mode
    args_aishell4_wav_list = args.aishell4_wav_list
    args_textgrid_list = args.textgrid_list
    p_lst = []
    for i in range(num_process):
        p = Process(target=run, args=(wavlist, noiselist, rirlist, output_dir, datamode,args_text, args_mode,args_aishell4_wav_list,args_textgrid_list))
        p.start()
        p_lst.append(p)
