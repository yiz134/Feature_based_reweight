import torch
import torch.nn as nn
from tqdm import tqdm
from binary_experiments.utils import mean_center_K, distribution_K

def train_loop(net, criterion, train_X, train_y, val_X, val_y, optimizer, args, device):
    """
    Training loop with tqdm progress bar.
    
    Args:
        net: MLP model
        train_X: Training data
        train_y: Training labels
        val_X: Validation data
        val_y: Validation labels
        optimizer: Optimizer for training
        args: Arguments containing training parameters
        device: Device to train on
    """
    #criterion = nn.BCEWithLogitsLoss()
    
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Training loop
    for epoch in tqdm(range(args.num_epochs), desc="Training"):
        # Training phase
        net.train()
        optimizer.zero_grad()
        
        # Forward pass
        train_output = net(train_X)
        
        if args.visualize_type == 'NTK' and (epoch % args.visualize_every == 0):
            print(epoch)
            print(epoch % args.visualize_every == 0)
            # Compute NTK matrix between training and validation samples
            with torch.set_grad_enabled(True):
                # Get indices for samples with labels 1 and -1
                train_y1_indices = torch.where(train_y == -1)[0]
                train_y2_indices = torch.where(train_y == 1)[0]
                val_y1_indices = torch.where(val_y == -1)[0]
                val_y2_indices = torch.where(val_y == 1)[0]
                
                # Ensure inputs are on the correct device
                train_X1_ntk = train_X[train_y1_indices][:500].to(device)
                train_X2_ntk = train_X[train_y2_indices][:500].to(device)
                val_X1_ntk = val_X[val_y1_indices][:100].to(device)
                val_X2_ntk = val_X[val_y2_indices][:100].to(device)
                NTK11 = net.compute_NTK(train_X1_ntk, val_X1_ntk)
                NTK12 = net.compute_NTK(train_X1_ntk, val_X2_ntk)
                NTK21 = net.compute_NTK(train_X2_ntk, val_X1_ntk)
                NTK22 = net.compute_NTK(train_X2_ntk, val_X2_ntk)
                NTK = torch.cat([torch.cat([NTK11, NTK12], dim=1), torch.cat([NTK21, NTK22], dim=1)], dim=0)
                NTK_centered = NTK#mean_center_K(NTK)
                distribution_K(NTK_centered, 500, 100)
        elif args.visualize_type == 'last_layer_feature':
            pass

        train_loss = criterion(train_output, train_y)
        
        # Backward pass
        train_loss.backward()
        optimizer.step()
        
        # Compute training accuracy
        train_preds = torch.sign(train_output).int()
        train_acc = (train_preds == train_y.int()).float().mean()
        print(train_acc)
        
        # Validation phase
        net.eval()
        with torch.no_grad():
            val_output = net(val_X)
            val_loss = criterion(val_output, val_y)
            val_preds = torch.sign(val_output).int()
            val_acc = (val_preds == val_y.int()).float().mean()
        
        # Store metrics
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        train_accs.append(train_acc.item())
        val_accs.append(val_acc.item())
        
        # Update progress bar description
        tqdm.write(f"Epoch {epoch+1}/{args.num_epochs}")
        tqdm.write(f"Train Loss: {train_loss.item():.4f}, Train Acc: {train_acc.item():.4f}")
        tqdm.write(f"Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc.item():.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }