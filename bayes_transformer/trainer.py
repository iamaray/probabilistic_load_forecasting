import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import datetime
import os

from .model import MyModel
from torch.utils.tensorboard import SummaryWriter
from .utils import loadDataforTrainVal


class Trainer:
    def __init__(self, input_size=24*3, output_size=1, batch_size=64, epochs=100,
                 num_targets=3, num_aux_feats=13, window_len=72, ahead=1):
        # GPU setup
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("using GPU")
        else:
            print("using CPU")

        # Data loading
        x_train, y_train = loadDataforTrainVal(
            input_size=input_size, output_size=output_size)
        train_dataset = TensorDataset(x_train, y_train)
        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=batch_size, shuffle=True)
        self.Nbatch = len(self.train_loader)

        # Model setup
        self.net = MyModel(ahead=ahead,
                           num_targets=num_targets,
                           num_aux_feats=num_aux_feats,
                           window_len=window_len)
        self.net.to(self.device)
        print('    Total params: %.2fM' % (np.sum(p.numel()
              for p in self.net.parameters()) / 1000000.0))

        # Training setup
        self.writer = SummaryWriter('logfiles')
        self.savepath = 'modelsave/bmdet/'
        if not (os.path.exists(self.savepath)):
            os.makedirs(self.savepath)

        self.epochs = epochs
        self.num_targets = num_targets
        self.loss_train = np.zeros((epochs, num_targets))
        self.mu_list = np.zeros((epochs, num_targets))
        self.rho_list = np.zeros((epochs, num_targets))

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

                overall_loss, losses, p_mu, p_rho = self.net.fit(
                    x=inputs, y=labels, samples=ELBO_samples)

                self.mu_list[epoch] += p_mu.cpu().detach().numpy()
                self.rho_list[epoch] += p_rho.cpu().detach().numpy()
                print(p_mu.cpu().detach().numpy())
                print(p_rho.cpu().detach().numpy())

                for j in range(self.num_targets):
                    self.loss_train[epoch, j] += losses[j]
                nb_samples += len(inputs)

                end_time = datetime.datetime.now()
                lasttime = (end_time - start_time) * (self.Nbatch - i) + (end_time - start_time) * self.Nbatch * (
                    self.epochs - epoch - 1)

                loss_str = " ".join(
                    [f"loss{j+1}: {losses[j].item():.4f}" for j in range(self.num_targets)])
                print(f" eta: {lasttime} epoch: {epoch+1:4d} in {self.epochs:4d}, batch: {i+1:5d} "
                      f"loss: {overall_loss.item():.4f} LossChange: {overall_loss.item() - lastloss:.4f} {loss_str}")

                lastloss = overall_loss.item()

                self._log_metrics(overall_loss, losses, p_mu, p_rho, epoch, i)

            self.loss_train[epoch] = self.loss_train[epoch] / self.Nbatch
            self.mu_list[epoch] = self.mu_list[epoch] / self.Nbatch
            self.rho_list[epoch] = self.rho_list[epoch] / self.Nbatch

        self._save_model()
        print('Finished Training')

    def _log_metrics(self, overall_loss, losses, p_mu, p_rho, epoch, batch):
        step = self.Nbatch * epoch + batch
        self.writer.add_scalar('loss', overall_loss.item(), step)

        for i in range(self.num_targets):
            self.writer.add_scalar(f'loss{i+1}', losses[i].item(), step)

        p_mu_np = p_mu.cpu().detach().numpy()
        p_rho_np = p_rho.cpu().detach().numpy()
        for i in range(self.num_targets):
            self.writer.add_scalar(f'p_mu{i}', p_mu_np[i].item(), step)
            self.writer.add_scalar(f'p_rho{i}', p_rho_np[i].item(), step)

    def _save_model(self):
        torch.save(self.net, self.savepath + 'bmdet.pt')
        np.save(self.savepath + "mu_list", np.asarray(self.mu_list))
        np.save(self.savepath + "rho_list", np.asarray(self.rho_list))
        np.save(self.savepath + "loss_train", np.asarray(self.loss_train))
