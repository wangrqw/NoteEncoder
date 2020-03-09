import os
import numpy as np
#NN packages
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import Autoencoder
from loader import Music, Normalize, ToTensor

#graphing packages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display

###################################Parameters###################################
from info import Info
transform = transforms.Compose([Normalize(), ToTensor()])
train_set = Music(Info.train_path,Info.train_arr, transform)
test_set = Music(Info.test_path,Info.test_arr, transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle = True)
test_loader1 = DataLoader(train_set, batch_size=128, shuffle = False)
test_loader2 = DataLoader(test_set, batch_size=128, shuffle = False)

# ######################################Test Start################################

# # trainloader
# test_mtx=[]
# for data in train_loader:
#     data = Variable(data).cpu()
#     data = data.tolist()
#     data = np.reshape(data, (-1,501))
#     print(np.max(data), np.min(data))
#     test_mtx.extend(data)
# test_mtx=np.array(test_mtx).T
# print(test_mtx.shape)
# plt.figure(dpi=500)
# librosa.display.specshow(test_mtx)
# plt.savefig('./test_mtx.pdf')
# quit()

# # # testloader
# # test_mtx=[]
# # for data in test_loader2:
# #     data = Variable(data).cpu()
# #     data = data.tolist()
# #     data = np.reshape(data, (-1,501))
# #     print(np.max(data), np.min(data))
# #     test_mtx.extend(data)
# # test_mtx=np.array(test_mtx).T
# # print(test_mtx.shape)
# # plt.figure(dpi=500)
# # librosa.display.specshow(test_mtx)
# # plt.savefig('./test_mtx.pdf')

# # # org
# # file_name = Info.test_path+Info.test_arr[0]
# # org = np.load(Info.test_path+Info.test_arr[0])
# # org = librosa.amplitude_to_db(org, ref=1.0)
# # plt.figure(dpi=500)
# # librosa.display.specshow(org)
# # plt.savefig('./org.pdf')
# ######################################Test End##################################

model = Autoencoder().cpu()
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)
num_epochs = Info.num_epochs
print(model)

####################################Training####################################
print("Start training {} epochs.".format(num_epochs))
loss_arr=[]
embed_mtx=[]
decode_mtx=[]

for epoch in range(num_epochs):
    for data in train_loader:
        data = Variable(data).cpu()
        _, output = model(data)
        # if epoch == num_epochs-1: #get decoded graph
        #     o = output.data.numpy()
        #     o = np.reshape(o,(o.shape[0],-1)).tolist()
        #     decode_mtx.extend(o)
        #     em = embed.data.numpy()
        #     em = np.reshape(em,(em.shape[0],-1)).tolist()
        #     embed_mtx.extend(em)
        loss = distance(data, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))
    loss_arr.append(loss.item())
    
# # test training data
# for data in test_loader1:
#     embed, output = model(data)
#     o = output.data.numpy()
#     o = np.reshape(o,(o.shape[0],-1)).tolist()
#     decode_mtx.extend(o)
#     em = embed.data.numpy()
#     em = np.reshape(em,(em.shape[0],-1)).tolist()
#     embed_mtx.extend(em)

# # test testing data
# for data in test_loader2:
#     data = Variable(data).cpu()
#     embed, output = model(data)
#     o = output.data.numpy()
#     o = np.reshape(o,(o.shape[0],-1)).tolist()
#     decode_mtx.extend(o)
#     em = embed.data.numpy()
#     em = np.reshape(em,(em.shape[0],-1)).tolist()
#     embed_mtx.extend(em)

###################################Save Model###################################
os.system('mkdir {}'.format(Info.outpath))
torch.save(model.state_dict(), Info.outpath+'model.pth')
#save loss graph
np.save(Info.outpath+'loss_arr', loss_arr)
# #save decode graph
# decode_mtx = np.transpose(np.array(decode_mtx))
# print('decode_mtx', decode_mtx)
# np.save(Info.outpath+'decode_mtx', decode_mtx)
# #save latent variable
# embed_mtx = np.transpose(np.array(embed_mtx))
# print('embed_mtx', embed_mtx)
# np.save(Info.outpath+'embed_mtx', embed_mtx)