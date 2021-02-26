import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class Model(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, bidirection, number_layers, ks):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.module_mol = nn.ModuleList() 
        self.module_mol.append(nn.Conv2d(input_size, 64, kernel_size= ks, padding= int((ks-1)/2)))
        self.module_mol.append(nn.BatchNorm2d(64))
        #self.module_mol.append(nn.Upsample([2,64]))
        self.module_mol.append(nn.Conv2d(64, 128, kernel_size= ks, padding= (int(ks/2+1), int((ks-1)/2))))
        self.module_mol.append(nn.BatchNorm2d(128))
        self.module_mol.append(nn.GRU(128, hidden_size, bidirectional= bidirection, batch_first=True, num_layers= number_layers))
        self.module_mol.append(nn.Upsample(64))

        for name, param in self.module_mol[4].named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

        if bidirection == True:
            self.input_fc = hidden_size*2
        else:
            self.input_fc = hidden_size

        self.fc1 = nn.Linear(self.input_fc, num_classes)

        

    def forward(self, input):
        self.bs = input.shape[0] 
        x = input.view(self.bs, input.shape[1], 1, input.shape[2])
        
        # Residual Conv1
        xi = input.transpose(1,2)
        xi = self.module_mol[5](xi).transpose(1,2)  
        xi = xi.view(self.bs, xi.shape[1], 1, xi.shape[2])

        # CONV1
        
        x = F.relu(self.module_mol[0](x))
        x = self.module_mol[1](x)
        
        # Residual Connection
        x_sum = x+xi

        # CONV2
        x = F.relu(self.module_mol[2](x_sum))
        x = self.module_mol[3](x)
        
        #RNN
        x = x.view(self.bs, x.shape[1], -1).transpose(1,2)
        x, _ = self.module_mol[4](x)
        
        # Classification
        x = x.transpose(0,1)[-1]
        x = self.fc1(x)

        return x
