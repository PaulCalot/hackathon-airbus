import torch
import numpy as np

class InferenceWrapper(torch.nn.Module):
    def __init__(self, classification, dv_net, time_net) -> None:
        super().__init__()
        self.classification = classification
        self.dv_net = dv_net
        self.time_net = time_net
    
    def forward(self, x):
        c, embedding = self.convnet.predict(x)
        dv = self.dv_net(embedding) 
        time = self.time_net(embedding)
        return (c, dv, time), embedding
    
    def predict(self, dataloader, use_convnet_embedding=False):
        self.eval()
        cc_list = []
        time_list = []
        dv_list = []

        time_true = []
        dv_list_true = []
        cc_list_true = []
        with torch.no_grad():
            for x, y in dataloader:
                c, embedding = self.classification(x)
                if(use_convnet_embedding):
                    dv = self.dv_net(embedding)
                    time = self.time_net(embedding)
                else:
                    dv = self.dv_net(x)
                    time = self.time_net(x)
                # saving preds
                cc_list.append(c.cpu().numpy())
                time_list.append(time.cpu().numpy())
                dv_list.append(dv.cpu().numpy())

                # saving preds
                time_true.append(y[2].cpu().numpy())
                dv_list_true.append(y[1].cpu().numpy())
                cc_list_true.append(y[0].cpu().numpy())
        return ((np.concatenate(cc_list),
                np.array(dv_list).flatten(),
                np.array(time_list).flatten()), 
                (np.concatenate(cc_list_true),
                 np.array(dv_list_true).flatten(),
                 np.array(time_true).flatten()))