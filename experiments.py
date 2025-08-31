from load_data.load_data import *
#from resnet import *
from trainer.train import *
from trainer.reweight_methods import *
import torch.optim as optim
from models import resnet
from models import model
import matplotlib.pyplot as plt
from load_data.data_utils import get_noisy_labels
import types
import os
from utils.util import *
from data_loader.cifar10 import load_cifar10

config = {
    "data_loader": {"args": {"data_dir": "G:/datasets"}},
    "trainer": {
        "percent": 0.4,
        "asym": True,
        "instance": False,
        "seed": 0,
    },
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train_loader, test_loader, X_val, y_val, train_clean_idx, train_noise_idx, val_idx = load_cifar((49000, 1000),
#                                                                                                 device='cuda',
#                                                                                                 noise_rate=0.5,
#                                                                                                 batch_size=128,
#                                                                                                 #noise_type='symmetric'
# )
train_loader, val_loader, test_loader, X_val, y_val, train_clean_idx, train_noise_idx = load_cifar10(config, 2000)
model = model.resnet34(num_classes=10)#resnet.resnet32()
model.reweight_method = types.MethodType(reweight_feature, model)
model = model.to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.02,#0.1
    momentum=0.9,
    weight_decay=0.001#1e-4
)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[40, 80, 100], last_epoch=- 1)#, try 50, gamma=0.001)  # 80, 100, total 120
w_list, loss = train_loop(model, optimizer, train_loader, X_val, y_val, test_loader,
                          alpha=[1/9, 0],#1/9
                          num_epochs=150,  start_epoch=0, val_feature=True,
                          reweight_every=1, max_clip=1, clean_only=False,
                          reweight=True, args=device, test_every=1, lr_scheduler=lr_scheduler,
                          noisy_rate=None, recompute=True, val_steps=30, skip_epochs=20, correction=None
                          )
# 100, 150, total 200
torch.save(w_list, "no_recompute.pth")
save_path = 'cifar_test_loss_results_noise=0.4'
if os.path.exists(save_path):
    data = torch.load(save_path)
else:
    data = {}

data['Not_Gradual'] = loss
torch.save(data, save_path)

# plot_weights_hist(w_list[-1].cpu(), train_clean_idx, train_noise_idx, val_idx)
# plot_weight_dynamic(w_list, train_clean_idx, train_noise_idx, val_idx)
# w_tensor = torch.stack(w_list, dim=0)
# w_tensor_max, _ = torch.max(w_tensor, dim=0)
# w_tensor_min, _ = torch.min(w_tensor, dim=0)
# w_gap = w_tensor_max - w_tensor_min
# target_idx = torch.where(w_gap >= 0.95)[0]
# train_dataset = train_loader.dataset
#
# images = train_dataset.images
# indices = train_dataset.indices
# pos = [torch.where(indices.cpu() == idx.cpu())[0][0] for idx in target_idx]
#
# # Get the i-th sample (img, y_clean, y_noise, idx)
# for i in range(10):
#     img = images[pos][i]
#     # If the image is a tensor (after transform), convert to numpy for plotting
#     if hasattr(img, 'numpy'):
#         # Undo normalization for visualization
#         mean = [0.4914, 0.4822, 0.4465]
#         std = [0.2470, 0.2435, 0.2616]
#         img_np = img.numpy().transpose(1, 2, 0) * std + mean
#         img_np = img_np.clip(0, 1)
#     else:
#         img_np = np.array(img)
#
#     plt.imshow(img_np)
#     #plt.title(f"Index: {i}, Clean label: {y_clean}, Noisy label: {y_noise}")
#     plt.axis('off')
#     plt.show()