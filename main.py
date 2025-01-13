from preprocessing import readtoFiltered, preprocess
from bayes_transformer.model import BSMDeTWrapper
from bayes_transformer.trainer import BayesTrainer
from grid_search import grid_search_torch_model

df, train_loader, test_loader = preprocess(
    'data/upto_latest_actual.csv', variates=[])
model_wrapper = BSMDeTWrapper(cuda=False)
trainer = BayesTrainer(model_wrapper=model_wrapper, train_loader=train_loader)

# trainer.train()
trainer.test(test_loader=test_loader)

# for x, y in train_loader:
#     x = x.transpose(1, 2)
#     y = y.transpose(1, 2) 

#     o = model_wrapper.model(x)
#     print(o.shape)

param_grid = {
    'd_model': [16, 32, 64, 128, 256],
    'encoder_layers': [2, 3, 4],
    'encoder_d_ff': [64, 128, 256],
    'encoder_sublayers': [2, 3],
    'encoder_h': [8, 16, 32],
    'encoder_dropout': [0.1],
    'decoder_layers': [2, 3, 4],
    'decoder_dropout': [0.1, 0.2],
    'decoder_h': [8, 16, 32],
    'decoder_d_ff': [64, 128, 256],
    'decoder_sublayers': [3, 4],
}
training_args = {'epochs': 1}

# grid_search_torch_model(model_class=BSMDeTWrapper, trainer_class=BayesTrainer,
#                         param_grid=param_grid, training_args=training_args, train_loader=train_loader, test_loader=test_loader)
