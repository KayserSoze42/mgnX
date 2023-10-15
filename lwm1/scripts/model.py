import torch.nn as nn
from torch.nn.models import resnet50, ResNet50_Weights

class LWM1CNN(nn.Module):

    def __init__(self, num_mush_classes, num_surr_classes):

        super(LWM1CNN, self).__init__() # still u af, lol

        # baseg
        self.baseg = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) # deprecated=True, mhm
        
        # remove layers - the final countdown - !whitestripes, lol
        self.features = nn.Sequential(*list(self.baseg.children())[:-1])

        # fine, just fine
        for param in list(self.baseg.parameters())[:-6]:
            param.requires_grad = False

        # eraserheads

        # mush
        self.mush_head = nn.Sequential(

                nn.Linear(2048, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_mush_classes)

        )

        # surr
        self.surr_head = nn.Sequential(

                nn.Linear(2048, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_surr_classes)

        )

    def forward(self, x):

        # Xtr-resnet
        x = self.features(x)
        x = x.view(x.size(0), -1) # 1deez these deezes

        mush_out = self.mush_head(x)
        surr_out = self.surr_head(x)

        return mush_out, surr_out

def setup_model(num_mush_classes, num_surr_classes):
    model = LWM1CNN(num_mush_classes, num_surr_classes)
    return model
