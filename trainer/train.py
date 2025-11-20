import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils.util import *
from trainer.utils import *
from tqdm import tqdm
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_features_batched(model, X, batch_size=256):
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    feats_list = []
    device = X.device
    with torch.no_grad():
        for (xb,) in loader:
            f = model.get_features(xb)
            feats_list.append(f.detach().cpu())
    features = torch.cat(feats_list, dim=0).to(device)
    return features

def train_loop(model, optimizer,
    train_loader, X_val, y_val, test_loader,
    alpha=[0.1, 0], visualize=False, start_epoch=0,
    num_epochs=350, val_steps=None, skip_epochs=None,
    reweight_every=5, correction=None, max_clip=1, clean_only=False, val_feature=False,
    reweight=False, args=None, test_every=5, lr_scheduler=None, noisy_rate=None, recompute=True, imbalance=False):

    # Initialize optimizer and loss function
    criterion = nn.CrossEntropyLoss(reduction='none')
    if start_epoch > 0:
        model, optimizer, lr_scheduler, _, loss = load_checkpoint(model, optimizer, lr_scheduler, path="checkpoint_90.pth")
    # Initialize sample weights and weight history
    sample_weights = torch.ones(len(train_loader.dataset), device=args) / 2
    weight_list = [sample_weights.clone().detach()]  # Record initial weights
    test_acc_list = []
    visualize_list = [10, 60, 80, 100, 120, 140]
    # if val_steps > 0 and start_epoch==0:
    #     train_steps(model, X_val, y_val, criterion, optimizer, val_steps)
    #     test_correct = 0
    #     test_total = 0
    #     with torch.no_grad():
    #         for data, _, _, target in test_loader:
    #             data = data.to('cuda')
    #             target = target.to('cuda')
    #             output = model(data)
    #             pred = output.argmax(dim=1)
    #             test_correct += pred.eq(target).sum().item()
    #             test_total += target.size(0)
    #     test_acc = test_correct / test_total
    #     print(f'1st Stage Test Accuracy: {100.*test_acc:.2f}%')

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        if epoch<40:
            alpha0 = alpha[0]
        else:
            alpha0 = alpha[1]
        #save_checkpoint(model, optimizer, epoch, loss, lr_scheduler, path="checkpoint.pth")
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        if val_steps > 0:
            train_steps(model, X_val, y_val, criterion, optimizer, val_steps)
        # Create progress bar for batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch")
        if val_feature:
            model.eval()
            val_input = get_features_batched(model, X_val)#model.get_features(X_val)
            model.train()
        else:
            val_input = X_val
        for data, target, indices, y_clean in pbar:
            # Update sample weights if reweighting is enabled
            data = data.cuda()
            target = target.cuda()
            if reweight and epoch > skip_epochs and epoch % reweight_every == 0:
                model.eval()
                with torch.no_grad():
                    if epoch in visualize_list:
                        visualize = True
                        visualize_list.pop(0)
                    else:
                        visualize = False
                    sample_weights[indices], Kmat, target, Kmat_raw = model.reweight_method(sample_weights[indices], data, target,
                                                                    val_input, y_val, alpha0=alpha0,
                                                                    recompute=recompute, ratio=noisy_rate,
                                                                    max_clip=max_clip, visualize=visualize,
                                                                    val_feature=val_feature)
                    # if epoch >= 83:
                    #     batch_weights = sample_weights[indices]
                    #     plot_idx = get_noisy_high_weight_idx(batch_weights, target, y_clean,
                    #                                          idx_type='noisy_negative')
                    #     plot_idx = plot_idx[:10]
                    #     plot_value_distributions(Kmat_raw[plot_idx], target[plot_idx], y_clean[plot_idx], torch.arange(10, device=y_val.device, dtype=y_val.dtype), num_class=int(y_val.max()+1), mean=True)
                model.train()

            output = model(data)
            
            batch_weights = sample_weights[indices]
            if clean_only:
                mask = (y_clean.cuda() == target)
                batch_weights[mask] = 1.0
                batch_weights[~mask] = 0.0
            batch_noisy_rate = get_noisy_rate(batch_weights, y_clean, target)
            if imbalance:
                batch_weights = get_balanced_weight(batch_weights.detach().clone(), target)
            loss = (criterion(output, target) * batch_weights.detach().clone()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            pbar.set_postfix(
                batch_noisy_rate=f"{100*batch_noisy_rate:5.2f}%",
                loss=f"{loss.item():.4f}",
                acc=f"{100 * correct / total:5.2f}%",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}"
            )

        if lr_scheduler is not None:
            lr_scheduler.step()

        if reweight and epoch > skip_epochs and  epoch % reweight_every == 0:
            weight_list.append(sample_weights.clone().detach())  # Record updated weights
            
        # Calculate test accuracy every test_every epochs
        if epoch % test_every == 0:
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for data, _, _, target in test_loader:
                    data = data.to('cuda')
                    target = target.to('cuda')
                    output = model(data)
                    pred = output.argmax(dim=1)
                    test_correct += pred.eq(target).sum().item()
                    test_total += target.size(0)
            
            test_acc = test_correct / test_total
            test_acc_list.append(test_acc)
            print(f'Epoch {epoch+1} Test Accuracy: {100.*test_acc:.2f}%')
    # if val_steps > 0:
    #     for val_step in tqdm(range(val_steps)):
    #         model.train()
    #         # Training on validation data
    #         output = model(X_val)
    #         loss = criterion(output, y_val).mean()
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #     model.eval()
    #     test_correct = 0
    #     test_total = 0
    #     with torch.no_grad():
    #         for data, target in test_loader:
    #             data = data.to('cuda')
    #             target = target.to('cuda')
    #             output = model(data)
    #             pred = output.argmax(dim=1)
    #             test_correct += pred.eq(target).sum().item()
    #             test_total += target.size(0)
    #     test_acc = test_correct / test_total
    #     print(f'Final Stage Test Accuracy: {100. * test_acc:.2f}%')
    # Final evaluation on test set
    model.eval()
    # test_correct = 0
    # test_total = 0
    # with torch.no_grad():
    #     for data, target in tqdm(test_loader, desc='Testing'):
    #         data = data.cuda()
    #         target = target.cuda()
    #         output = model(data)
    #         pred = output.argmax(dim=1)
    #         test_correct += pred.eq(target).sum().item()
    #         test_total += target.size(0)
    #
    # test_acc = test_correct / test_total
    # test_acc_list.append(test_acc)  # Add final test accuracy
    # print(f'Final Test Accuracy: {100.*test_acc:.2f}%')
    #
    return weight_list, test_acc_list

def train_steps(model, X, y, criterion, optimizer, num_steps,
                batch_size=128, shuffle=True, num_workers=0):
    """
    Mini-batch training where `num_steps` = number of epochs.

    Args:
        model: nn.Module
        X (Tensor): inputs, shape [N, ...]
        y (Tensor): targets, shape [N]
        criterion: loss fn; if it returns per-sample loss, we .mean() it
        optimizer: torch optimizer
        num_steps (int): epochs
        batch_size (int): mini-batch size (default 256)
        shuffle (bool): shuffle each epoch
        num_workers (int): dataloader workers
    """
    device = next(model.parameters()).device

    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        drop_last=False, num_workers=num_workers)

    model.train()
    for epoch in tqdm(range(num_steps), desc="Epochs"):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            out = model(xb)
            loss = criterion(out, yb).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()