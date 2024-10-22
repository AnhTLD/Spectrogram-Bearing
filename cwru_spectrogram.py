from multiprocessing import Pool
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import os

CWRU = True

if CWRU:
    namefolder = r'C:\Users\khoiv\OneDrive\OneDrive - Hanoi University of Science and Technology\Desktop\CWRU data'
    NAMEFILE = ['109', '122', '135',
               '110','123','136',
               '111','124','137',
               '112','125','138',
               '213','226','238',
               '214','227','239',
               '215','228','240',
               '217','229','241',               
               '97','98','99','100'
               ]
    sampleRate = 48000
else:
    namefolder = r'C:\Users\khoiv\OneDrive\OneDrive - Hanoi University of Science and Technology\Desktop\HUST data'
    NAMEFILE = ['B500', 'I500', 'N500', 'O500']
    sampleRate = 51200

def create_and_save_figure(namefile,f,t,stft_data,i):
    #fig = plt.figure(figsize=(480/100, 240/100))
    plt.pcolormesh(t,
                   f,
                   np.abs(stft_data),
                   vmin=np.min(np.abs(stft_data)),
                   #vmax=np.max(np.abs(stft_data)),
                   vmax=10,
                   shading='gouraud',
                   cmap='viridis')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(namefolder+'\\'+namefile, f'{namefile[-1]}({i}).png'))
    plt.clf()
    #plt.close(fig)


def process_file(namefile):

    mat = loadmat(namefolder+'\\'+ namefile + '.mat')

    if CWRU:
        if len(namefile) < 3:
            namefile = '0' + namefile
        data = mat['X' + namefile + '_DE_time']
        data = data.reshape(-1)
        if namefile == '098':
             RPM = 1772
        elif namefile == '099':
             RPM = 1750
        else:
            RPM = mat['X' + namefile + 'RPM']

        num_round = sampleRate / (RPM / 60) # độ dài dữ liệu 1 vòng.

        num_overlapping = 1/4 # độ chồng lấn
        mum_data = 2 # độ dài 1 đoạn dữ liệu cắt

        num_overlapping = num_overlapping * num_round
        mum_data = num_round * mum_data
        
        # E07-0
        if namefile == '109':
            namefile = 'E07-0/I'
        elif namefile == '122':
            namefile = 'E07-0/B'
        elif namefile == '135':
            namefile = 'E07-0/O'

        # E07-1
        if namefile == '110':
            namefile = 'E07-1/I'
        elif namefile == '123':
            namefile = 'E07-1/B'
        elif namefile == '136':
            namefile = 'E07-1/O'

        # E07-2
        if namefile == '111':
            namefile = 'E07-2/I'
        elif namefile == '124':
            namefile = 'E07-2/B'
        elif namefile == '137':
            namefile = 'E07-2/O'

        # E07-3
        if namefile == '112':
            namefile = 'E07-3/I'
        elif namefile == '125':
            namefile = 'E07-3/B'
        elif namefile == '138':
            namefile = 'E07-3/O'

        # E21-0
        if namefile == '213':
            namefile = 'E21-0/I'
        elif namefile == '226':
            namefile = 'E21-0/B'
        elif namefile == '238':
            namefile = 'E21-0/O'

        # E21-1
        if namefile == '214':
            namefile = 'E21-1/I'
        elif namefile == '227':
            namefile = 'E21-1/B'
        elif namefile == '239':
            namefile = 'E21-1/O'

        # E21-2
        if namefile == '215':
            namefile = 'E21-2/I'
        elif namefile == '228':
            namefile = 'E21-2/B'
        elif namefile == '240':
            namefile = 'E21-2/O'

        # E21-3
        if namefile == '217':
            namefile = 'E21-3/I'
        elif namefile == '229':
            namefile = 'E21-3/B'
        elif namefile == '241':
            namefile = 'E21-3/O'

        # N00
        if namefile == '097':
            namefile = 'N00-0/N'
        elif namefile == '098':
            namefile = 'N00-1/N'
        elif namefile == '099':
            namefile = 'N00-2/N'
        elif namefile == '100':
            namefile = 'N00-3/N'
        
            
    else: #hust
        data = mat['data']
        data = data.reshape(-1)
        fs = mat['fs']
        RPM = fs * 60

        num_round = sampleRate / (RPM / 60) # độ dài dữ liệu 1 vòng.

        num_overlapping = 1/3 # độ chồng lấn
        mum_data = 3 # độ dài 1 đoạn dữ liệu cắt

        num_overlapping = num_overlapping * num_round
        num_round = num_round * mum_data
    
        if namefile == 'N500':
            namefile = 'Target/N'
        elif namefile == 'I500':
            namefile = 'Target/I'
        elif namefile == 'B500':
            namefile = 'Target/B'
        elif namefile == 'O500':
            namefile = 'Target/O'


    os.makedirs(namefolder+'\\'+namefile, exist_ok=True)
    print(namefile + ' Start') 
    for i in range(0, int((data.shape[0] - mum_data)/num_overlapping), 1):
        f, t, stft_data = signal.stft(
            data[int(i * num_overlapping):int(i * num_overlapping + mum_data)],
            window='hann',
            fs=sampleRate,
            nperseg=int(RPM/36),
            nfft= int(RPM/36) * 2
            )
        create_and_save_figure(namefile,f,t,stft_data,i)  
        
    print(namefile + ' Done\n')   


if __name__ == '__main__':
    pool = Pool()
    pool.map(process_file, [(namefile) for namefile in NAMEFILE])
    pool.close()
    pool.join()