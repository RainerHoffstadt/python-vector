import torch.nn as nn
import torchvision.models as models


#ChessClasses = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']
ChessClasses = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'empty', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']
transfer_model = models.resnet50(pretrained=True)
for name, param in transfer_model.named_parameters():
    #if("bn" not in name):
        param.requires_grad = False

transfer_model.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features, 520),
                                  nn.ReLU(), nn.Dropout(), nn.Linear(520, 13))