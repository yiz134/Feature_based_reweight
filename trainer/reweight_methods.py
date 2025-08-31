import torch
from trainer.utils import *
from utils.util import *
def reweight_ntk(self, weight, X_tr, y_tr, X_val, y_val):
    pass


def reweight_feature(self, weight, X_tr, y_tr, X_val, y_val,
                     alpha0=1, recompute=False, ratio=None, max_clip=1, visualize=False, val_feature=False):
    num_classes = int(y_val.max())+1
    tr_features = self.get_features(X_tr) # shape: (N_tr, D)
    if val_feature:
        val_features = X_val
    else:
        val_features = self.get_features(X_val) # shape: (N_val, D)
    mean_feature = weighted_mean_features(val_features, y_val) # shape: (1, D)
    tr_features = tr_features - mean_feature
    val_features = val_features - mean_feature

    # tr_features = tr_features / tr_features.norm(dim=1, keepdim=True)
    # val_features = val_features / val_features.norm(dim=1, keepdim=True)

    Kmat = tr_features @ val_features.T    # shape: (N_tr, N_val)
    Kmat, y_val, Kmat_raw = transform_Kmat(Kmat, y_val, num_class=num_classes)
    #Kmat = tr_norm @ val_norm.T
    # K_val_class_wise = get_K_val_class_wise(val_features, y_val)
    # diag_min = K_val_class_wise.diagonal().min()
    #Kmat = Kmat - 0.5 * diag_min
    if visualize:
        K_val = val_features @ val_features.T #- 0.5 * diag_min
        visualize_Kmat(K_val, y_val)

    match = (y_tr[:, None] == y_val[None, :])      # (N_tr, N_val), dtype=bool
    sign_mask = torch.where(match, 
                            torch.tensor(1.0, device=Kmat.device), 
                            torch.tensor(-alpha0, device=Kmat.device))  # (N_tr, N_val)

    # Apply sign to kernel
    Kmat_signed = Kmat * sign_mask #/10  # shape: (N_tr, N_val)
    w_direction = torch.sum(Kmat_signed, dim=1) # shape: (N_tr,)

    if recompute:
        weight = torch.ones_like(weight, device=weight.device) /2 #10#(0.1*len(weight))
        weight = update_weight(weight, w_direction, clipmax=max_clip, ratio=ratio)
    else:
        weight = update_weight(weight, w_direction, clipmax=max_clip, ratio=ratio, eta=0.01)

    weight[weight <= 0.5] = 0
    return weight, Kmat, y_tr, Kmat_raw


def reweight_ntk(self, weight, X_tr, y_tr, X_val, y_val, alpha0=1, recompute=False, ratio=None):
    tr_features = self.get_tangent_features(X_tr, y_tr)  # shape: (N_tr, D)
    val_features = self.get_tangent_features(X_val, y_val)  # shape: (N_val, D)
    mean_feature = weighted_mean_features(val_features, y_val)  # shape: (1, D)
    tr_features = tr_features - mean_feature
    val_features = val_features - mean_feature

    Kmat = tr_features @ val_features.T  # shape: (N_tr, N_val)

    match = (y_tr[:, None] == y_val[None, :])  # (N_tr, N_val), dtype=bool
    sign_mask = torch.where(match,
                            torch.tensor(1.0, device=Kmat.device),
                            torch.tensor(-alpha0, device=Kmat.device))  # (N_tr, N_val)

    # Apply sign to kernel
    Kmat_signed = Kmat * sign_mask  # shape: (N_tr, N_val)
    w_direction = torch.sum(Kmat_signed, dim=1)  # shape: (N_tr,)
    if recompute:
        weight = torch.ones_like(weight, device=weight.device) / len(weight)
        weight = update_weight(weight, w_direction, clipmax=1)
    else:
        weight = update_weight(weight, clipmax=1)
    return weight




    
