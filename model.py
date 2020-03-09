import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(501, 128)
            ,nn.LeakyReLU(True)
            ,nn.Linear(128, 32)
            ,nn.LeakyReLU(True)
            # ,nn.Linear(32, 16)
            # ,nn.LeakyReLU(True)
            # nn.Linear(32, 16),
            # nn.LeakyReLU(True)
        )
        self.decoder = nn.Sequential(
            # nn.Linear(16, 32),
            # nn.LeakyReLU(True),
            # nn.Linear(16, 32),
            # nn.LeakyReLU(True),
            nn.Linear(32, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, 501),
            nn.Sigmoid()
        )
            
    def forward(self,x):
        x = self.encoder(x)
        y = self.decoder(x)
        return x, y

################################### Old Version ################################
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder,self).__init__()
        
#         # self.act = nn.ReLU(True)
#         self.act = nn.LeakyReLU(True)
#         self.l1 = nn.Linear(501, 32)
#         self.l2 = nn.Linear(32, 32)
#         self.l3 = nn.Linear(32, 32)
#         self.l4 = nn.Linear(32, 16)
#         self.l4T = nn.Linear(16, 32)
#         self.l3T = nn.Linear(32, 32)
#         self.l2T = nn.Linear(32, 32)
#         self.l1T = nn.Linear(32, 501)
#         self.sigmoid = nn.Sigmoid()
            
#     def forward(self,x):
#         # Encoder
#         # print("input size", x.size())
#         # print(x)
#         x = self.l1(x)
#         x = self.act(x)
#         # print('l1 size', x.size())
#         x = self.l2(x)
#         x = self.act(x)
#         # print('l2 size: ', x.size())
#         x = self.l3(x)
#         x = self.act(x)
#         # print('l3 size: ', x.size())
#         x = self.l4(x)
#         x = self.act(x)
        
#         # Decoder
#         y = self.l4T(x)
#         y = self.act(y)
#         # print('l4T size: ', y.size())
#         y = self.l3T(y)
#         y = self.act(y)
#         # print("l3T size ", y.size())
#         y = self.l2T(y)
#         y = self.act(y)
#         # print("l2T ", y.size())
#         y = self.l1T(y)
#         y = self.sigmoid(y)
