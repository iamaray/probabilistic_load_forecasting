from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim


def model_saver(model):
    def save_model(self):
        torch.save(nn_class.state_dict(),
                   f"/saved_models/{self.model_name}_weights_{datetime.now().time()}.pth")
    setattr(nn_class, "save_model", save_model)
