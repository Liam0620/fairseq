rirlist='/data08/home/fanzhiyun/program/aishell4/AISHELL-4-master/asr/data/rir_wavlist'
noiselist='/data08/home/fanzhiyun/program/aishell4/AISHELL-4-master/asr/data/rir_wavlist'
aishell1='/data08/home/fanzhiyun/data/aishell1/process/train_tmp/wavlist'
text='/data08/home/fanzhiyun/data/aishell1/process/train_tmp/text'
outputdir='/data08/home/fanzhiyun/program/aishell4/AISHELL-4-master/asr/data'
aishell4text='/data08/home/fanzhiyun/data/aishell4/process/train/aishell4_train_textgridlist'
aishell4wav='/data08/home/fanzhiyun/data/aishell4/process/train/aishell4_train_wavlist'

python -u asr_data_noise_rir.py --aishell1_wav_list $aishell1 --text $text --noise_list $noiselist --rir_list $rirlist --output_dir $outputdir --mode train --aishell4_wav_list $aishell4wav --textgrid_list $aishell4text 

