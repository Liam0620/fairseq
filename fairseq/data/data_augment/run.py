from speech_perturb import SpeedPerturb
from  asr_data_noise_rir import add_noise_rir ,NoiseRIR_Dataset
from volume_perturb import volume_perturb
import soundfile as sf
import pdb
import torch as th

source_wav = '/data/LibriSpeech/example.wav'
rir_path = '/data/LibriSpeech/RIRS_NOISES/real_rirs_isotropic_noises/air_type1_air_binaural_stairway_1_3_60.wav'
noise_path = '/data/LibriSpeech/noise.wav'

samples, sr = sf.read(source_wav)
print(1111,samples.shape)

#add noise rir
noise_rir_dataset = NoiseRIR_Dataset( '/workspace/fairseq/manifest/augmentation/noises.txt','/workspace/fairseq/manifest/augmentation/rirs.txt',5,20)
samples = noise_rir_dataset.add_noise_rir(source_wav)
#samples = add_noise_rir(source_wav, noise_path, rir_path)

#pdb.set_trace()
print(22222,samples.shape)
# add speed perturb
sp = SpeedPerturb(sr= 16000, perturb= "0.9,1.1")
# pdb.set_trace()
samples = th.tensor(samples)

samples = samples.unsqueeze(0)
print(33333,samples.size())
# input a 2-D tensor
out = sp(samples)
print(31233434,out.size())
# add volume perturb
out = volume_perturb(out,low=-1.6,high=1.6)
# print(3213123,out)
# print(4444,out.size())
print('done!')


