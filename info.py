import numpy as np
import os
import librosa

def get_max_min(path,file_arr,extention='_mag.npy'):
    # file_arr = [[1,2], [3,4]]
    # mtx = file_arr[0]
    # print('file_arr:',file_arr)
    # print(path+file_arr[0]+extention)
    mtx = np.load(path+file_arr[0]+extention)
    mtx = librosa.amplitude_to_db(mtx, ref=1.0)
    maxi = np.max(mtx)
    mini = np.min(mtx)
    for i in range(1,len(file_arr)):
        # mtx = file_arr[i]
        mtx = np.load(path+file_arr[i]+extention)
        mtx = librosa.amplitude_to_db(mtx, ref=1.0)
        maxi = max(maxi, np.max(mtx))
        mini = min(mini, np.min(mtx))
    return maxi,mini

def get_arr_from_folder(path):
    file_arr=[]
    for root,dirs,files in os.walk(path):
        files = np.sort(np.array(files))
        # for f in files:
        #     if not f.startswith('.') and f.endswith('.npy'):
        #         file_arr.append(f)
        for i in range(files.shape[0]//64):
            arr = files[i*64:(i+1)*64]
            file_arr.extend(np.random.choice(files,10).tolist())
        # print(file_arr)
    return file_arr
        
# def get_music_arr(inpath,file_arr):
#     music_arr = []
#     for i in range(len(mag_arr)):
#         mag_file = inpath+mag_arr[i]
#         music_arr.append(np.load(mag_file))
#     return music_arr

# def get_mag_arr(file_arr):
#     mag_arr=[]
#     for i in range(len(file_arr)):
#         mag_arr.append(file_arr[i]+'_mag.npy')
#     return mag_arr

# def get_phs_arr(file_arr):
#     phs_arr=[]
#     for i in range(len(file_arr)):
#         phs_arr.append(file_arr[i]+'_phs.npy')
#     return phs_arr

class Info():
    root='/users/PAS0027/wang6863/Desktop/NoteEncoder/'
    train_path=root+'Data/chunk/'
    test_path=root+'Data/'
    # train_arr=['D1_e_mag.npy', 'D1_z_mag.npy']
    train_arr=get_arr_from_folder(train_path)
    # test_arr=['09LiangXiao_mag.npy']
    test_arr=['01GuanShanXing_mag.npy']
    maxi,mini = get_max_min(train_path, train_arr, '')
    # print(maxi, mini)
    outpath=root+'Output/{}/'.format('sample128_32batch128')
    num_epochs = 500
    
    