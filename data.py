from data_loader.cifar10 import CIFAR10DataLoader  # 就是你这个文件定义的类

# 1) 配置（关键只要这几个字段）
config = {
    "data_loader": {"args": {"data_dir": "G:/datasets"}},   # 改成你的路径
    "trainer": {
        "percent": 0.2,     # 噪声比例：这里期待的是 0~1（例如 0.4 表示 40%）
        "asym": False,      # True=非对称噪声；False 时走 symmetric_noise()
        "instance": False,  # True=实例相关噪声；优先级高于 asym
        "seed": 0,
    },
}

# 2) 构建带噪声的 DataLoader（90% 训练 + 10% “验证集”，两者都会按上面的策略加噪）
train_loader = CIFAR10DataLoader(
    data_dir=config["data_loader"]["args"]["data_dir"],
    batch_size=256,
    shuffle=True,
    #validation_split=0.0,   # 你这里内部已经做了 9:1 的划分
    num_workers=0,
    training=True,          # 训练阶段 -> 会对 train/val 都加噪
    config=config,
    seed=config["trainer"]["seed"],
)

val_loader = train_loader.split_validation()

# 3) 训练时使用（注意 __getitem__ 返回四元组）
for imgs, noisy_y, idx, clean_y in train_loader:
    print(imgs.size())
    print(noisy_y)
    print(clean_y)
    print((noisy_y==clean_y).sum()/len(clean_y))
    break
    ...
