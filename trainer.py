import torch
from tqdm import tqdm
import numpy as np

class TorchTrainer:
    def __init__(self, model, verbose=True, weight=None, index_y=0, loss_function="BCELoss") -> None:
        self.model = model
        self.weight = weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        if(loss_function == 'CrossEntropyLoss'): # DO NOT USE IT
            loss_function = torch.nn.CrossEntropyLoss(weight=self.weight) # torch.tensor([0.05, 0.95]))
            self.loss_function = lambda pred, target: loss_function(pred, target.to(torch.float))
        elif(loss_function == "L1Loss"):
            self.loss_function = torch.nn.L1Loss()

        elif(loss_function == 'BCELoss'):
            loss_function =  torch.nn.BCEWithLogitsLoss(weight=self.weight)
            self.loss_function = lambda pred, target: loss_function(pred, target.to(torch.float))
        self.optimiser = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)
        self.verbose = verbose
        self.index_y = index_y
    
    def train(self, train_loader, epochs, lr):
        train_loss_list = []
        scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimiser, base_lr=lr/2., max_lr=2 * lr)
        t = tqdm(range(epochs), desc='0 - loss: 0', leave=True, disable=not self.verbose)
        for epoch in t:
            train_loss = self.train_one_epoch(train_loader, scheduler)
            train_loss_list.append(train_loss)
            t.set_description("{} - loss: {:0.2f}".format(epoch, train_loss), refresh=True)
        return train_loss_list

    def train_one_epoch(self, train_loader, scheduler):
        loss_list = []
        for x, y in train_loader:
            self.optimiser.zero_grad()
            x = x.to(self.device)
            y = tuple([y_.to(self.device) for y_ in y])
            pred, embedding = self.model(x)
            loss = self.loss_function(pred, torch.squeeze(y[self.index_y]))
            loss.backward()
            self.optimiser.step()
            loss_list.append(loss.detach().cpu().numpy())
            scheduler.step()

        return np.mean(loss_list)

    def predict(self, test_loader, return_true=False):
        self.model.eval()
        c_pred_list = []
        c_true_list = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = tuple([y_.to(self.device) for y_ in y])
                pred, embedding = self.model(x)
                c_pred_list.append(torch.squeeze(pred))
                # dv_pred_list.append(pred[1])
                c_true_list.append(torch.squeeze(y[self.index_y]))
                # dv_true_list.append(y[1])
        self.model.train()

        pred_tuple = (torch.concatenate(c_pred_list, axis=0),)
                # torch.concatenate(dv_pred_list, axis=0))        
        if(return_true):
            true_tuple = (torch.concatenate(c_true_list, axis=0),)
                # torch.concatenate(dv_true_list, axis=0))
            return true_tuple, pred_tuple
        return pred_tuple
