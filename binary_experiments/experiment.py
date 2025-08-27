import torch
import argparse
from binary_experiments.models import *
from binary_experiments.load_data import *
from binary_experiments.utils import *
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from binary_experiments.train import train_loop

def get_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs') #5000
    parser.add_argument('--hidden_dim', type=int, default=512, help='Dimension of hidden layer')
    parser.add_argument('--input_dim', type=int, default=28 * 28,
                        help='Dimension of input features (e.g., flattened image size)')
    parser.add_argument('--num_plot_pts', type=int, default=300, help='Number of points to plot')
    parser.add_argument('--k', type=int, default=50, help='Number of points to flip label')
    parser.add_argument('--weight_lr', type=float, default=0.0001, help='Learning rate for weights')  # 0.001
    parser.add_argument('--digit1', type=int, default=1, help='First digit to classify')
    parser.add_argument('--digit2', type=int, default=7, help='Second digit to classify')
    parser.add_argument('--label_scale', type=int, default=1)
    parser.add_argument('--one_hot', type=bool, default=False)
    parser.add_argument('--load_previous', type=bool, default=False)
    parser.add_argument('--visualize_type', type=str, default='NTK')
    parser.add_argument('--visualize_every', type=int, default=50)

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_X, train_y, val_X, val_y, test_X, test_y = prepare_mnist_binary_classification(args.digit1, args.digit2,
                                                                                         device='cuda')
    train_y = one_hot_to_signed_labels(train_y).unsqueeze(1)
    val_y = one_hot_to_signed_labels(val_y).unsqueeze(1)
    test_y = one_hot_to_signed_labels(test_y).unsqueeze(1)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_X, train_y, val_X, val_y, test_X, test_y, shuffle=False)
    

    h_dims = [args.hidden_dim, args.hidden_dim, args.hidden_dim]
    output_dim = 1
    net = MLP(args.input_dim, h_dims, output_dim=output_dim).to(device)
    
    with torch.no_grad():
        net.eval()
        init_output = net(train_X)
        hist_init_preds(init_output)
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate)
    loss = nn.MSELoss()
    metrics = train_loop(net, loss, train_X, train_y, val_X, val_y, optimizer, args, device)
