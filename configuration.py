import argparse
from tune_space import get_host_ip


def get_parameters():
    parser = argparse.ArgumentParser()

    # Initialize base_path first
    base_path = ''  # Default path if no server_ip or no path defined

    # Get server IP
    server_ip = get_host_ip()

    # If IP is available, update the base_path
    if server_ip != '':
        base_path = f'//{server_ip}/'  # Adjust path as needed

    parser.add_argument('--cmp', action='store_true', default=False, help="Enable comparison mode")

    # -- model parameters
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for regularization')

    # - embedding
    parser.add_argument('--emsize', type=int, default=300, help='Embedding dimension')

    # - lstm
    parser.add_argument('--nhid_lstm', type=int, default=516, help='LSTM hidden dimension')
    parser.add_argument('--nlayers_lstmEncoders', type=int, default=2, help='Number of LSTM encoder layers')

    # - transformer
    parser.add_argument('--d_model', type=int, default=300, help='The number of expected features in the input')
    parser.add_argument('--d_ffw', type=int, default=1024, help='Feedforward layer size')
    parser.add_argument('--nlayers_attnEncoders', type=int, default=4, help='Number of attention encoder layers')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')

    # - PBR
    parser.add_argument('--k', type=int, default=2, help='PBR hyperparameter K')
    parser.add_argument('--N', type=int, default=3, help='PBR hyperparameter N')
    parser.add_argument('--local_attn_dim', type=int, default=512, help='Local attention dimension')

    # - FFNN
    parser.add_argument('--nhid_ffnn', type=int, default=256, help='Hidden dimension of FFNN')

    # -- training settings
    parser.add_argument('--model_name', type=str, default='default', help='Name of the model')
    parser.add_argument('--model', type=str, default='default', help='Model architecture')  # Add --model argument
    parser.add_argument('--dataset', type=str, default='cn', choices=['cn', 'en'], help='Dataset choice')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'tune', 'pred'],
                        help='Mode of operation')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20000, help='Number of training epochs')
    parser.add_argument('--log_interval', type=int, default=50, help='Interval for logging')
    parser.add_argument('--print_wt', action='store_true', default=False, help='Flag to print weights')

    parser.add_argument('--use_l2', action='store_true', default=False, help='Flag to use L2 regularization')
    parser.add_argument('--reg_lambda', type=float, default=0.1, help='L2 regularization lambda')

    # -- data path / filename
    parser.add_argument('--path_text_processed', type=str, default=base_path + 'data/processed',
                        help='Path to processed text')
    parser.add_argument('--path_doc', type=str, default=base_path + 'data/doc/', help='Path to documents')
    parser.add_argument('--path_rawEngEleBase', type=str,
                        default=base_path + 'data/labelled_contracts/elements_contracts/',
                        help='Path to raw English element contracts')

    # - embeddings
    parser.add_argument('--path_emb', type=str, default=base_path + 'data/embeddings/', help='Path to embeddings')
    # - model parameters
    parser.add_argument('--path_model_para', type=str, default=base_path + 'data/model_para/',
                        help='Path to model parameters')

    return parser.parse_args()
