import imp


import torch


torch.load('./pre_train/LightCNN_29Layers_V2_checkpoint.pth.tar',map_location=torch.device('cpu'))
