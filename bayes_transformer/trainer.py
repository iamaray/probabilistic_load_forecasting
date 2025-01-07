import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import datetime
import os

from .model import MyModel
from torch.utils.tensorboard import SummaryWriter
from .utils import getDataforTrainVal

class Trainer:
    def __init__(self, input_size=24*3, output_size=1, batch_size=64, epochs=100):
        # GPU setup
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("using GPU")
        else:
            print("using CPU")
            
        # Data loading
        x_train, y_train = loadDataforTrainVal(input_size=input_size, output_size=output_size)
        train_dataset = TensorDataset(x_train, y_train)
        self.train_loader = DataLoader(dataset=train_dataset,
                                     batch_size=batch_size, shuffle=True)
        self.Nbatch = len(self.train_loader)
        
        # Model setup
        self.net = MyModel()
        self.net.to(self.device)
        print('    Total params: %.2fM' % (np.sum(p.numel()
              for p in self.net.parameters()) / 1000000.0))
              
        # Training setup
        self.writer = SummaryWriter('logfiles')
        self.savepath = 'modelsave/bmdet/'
        if not (os.path.exists(self.savepath)):
            os.makedirs(self.savepath)
            
        self.epochs = epochs
        self.loss_train = np.zeros((epochs, 3))
        self.mu_list = np.zeros((epochs, 3))
        self.rho_list = np.zeros((epochs, 3))

    def train(self):
        for epoch in range(self.epochs):
            if epoch == 0:
                ELBO_samples = 5
            else:
                ELBO_samples = 1

            nb_samples = 0
            lastloss = 0

            for i, data in enumerate(self.train_loader):
                start_time = datetime.datetime.now()

                inputs, labels = data
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device).float()

                overall_loss, loss1, loss2, loss3, p_mu, p_rho = self.net.fit(
                    x=inputs, y=labels, samples=ELBO_samples)

                self.mu_list[epoch] += p_mu.cpu().detach().numpy().reshape(3, )
                self.rho_list[epoch] += p_rho.cpu().detach().numpy().reshape(3, )
                print(p_mu.cpu().detach().numpy().reshape(3, ))
                print(p_rho.cpu().detach().numpy().reshape(3, ))

                self.loss_train[epoch, 0] += loss1
                self.loss_train[epoch, 1] += loss2
                self.loss_train[epoch, 2] += loss3
                nb_samples += len(inputs)

                end_time = datetime.datetime.now()
                lasttime = (end_time - start_time) * (self.Nbatch - i) + (end_time - start_time) * self.Nbatch * (
                    self.epochs - epoch - 1)
                print(" eta: ", lasttime,
                      " epoch: %4d in %4d, batch: %5d  loss: %.4f LossChange: %.4f  loss1: %.4f  loss2: %.4f  loss3: %.4f " % (
                          epoch + 1, self.epochs, (i + 1), overall_loss.item(
                          ), overall_loss.item() - lastloss, loss1.item(),
                          loss2.item(), loss3.item()))

                lastloss = overall_loss.item()

                self._log_metrics(overall_loss, loss1, loss2, loss3, p_mu, p_rho, epoch, i)

            self.loss_train[epoch] = self.loss_train[epoch] / self.Nbatch
            self.mu_list[epoch] = self.mu_list[epoch] / self.Nbatch
            self.rho_list[epoch] = self.rho_list[epoch] / self.Nbatch

        self._save_model()
        print('Finished Training')

    def _log_metrics(self, overall_loss, loss1, loss2, loss3, p_mu, p_rho, epoch, batch):
        step = self.Nbatch * epoch + batch
        self.writer.add_scalar('loss', overall_loss.item(), step)
        self.writer.add_scalar('loss1', loss1.item(), step)
        self.writer.add_scalar('loss2', loss2.item(), step)
        self.writer.add_scalar('loss3', loss3.item(), step)

        p_mu_np = p_mu.cpu().detach().numpy()
        p_rho_np = p_rho.cpu().detach().numpy()
        for i in range(3):
            self.writer.add_scalar(f'p_mu{i}', p_mu_np[i].item(), step)
            self.writer.add_scalar(f'p_rho{i}', p_rho_np[i].item(), step)

    def _save_model(self):
        torch.save(self.net, self.savepath + 'bmdet.pt')
        np.save(self.savepath + "mu_list", np.asarray(self.mu_list))
        np.save(self.savepath + "rho_list", np.asarray(self.rho_list))
        np.save(self.savepath + "loss_train", np.asarray(self.loss_train))
