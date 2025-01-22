from preprocessing import preprocess
from deepar.model import Net
from deepar.train import DeepARTrainer
from grid_search import grid_search_torch_model

if __name__ == "__main__":
    df, train_loader, test_loader, train_norm, test_norm = preprocess(
        csv_path='data/ercot_data_2025_Jan.csv',
        net_load_input=False,
        net_load_labels=True,
        auto_reg=True,
        variates=['marketday', 'ACTUAL_ERC_Load',
                  'ACTUAL_ERC_Solar', 'ACTUAL_ERC_Wind', 'hourending']
    )

    param_grid = {
        'num_class': [1],
        'embedding_dim': [1, 4, 16],
        'cov_dim': [4],
        'lstm_hidden_dim': [128, 256],
        'lstm_layers': [2, 3, 4],
        'lstm_dropout': [0.1],
        'predict_steps': [24],
        'predict_start': [0],
        'sample_times': [50, 100, 200],
        'device': ['cuda']
    }
