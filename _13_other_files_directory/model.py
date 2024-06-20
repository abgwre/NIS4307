import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from torchvision import models
from torchmetrics import Accuracy
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def load_images_from_folder(folder_path):
    img_list = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为 224x224
        transforms.ToTensor(),  # 将图像转换为Tensor，并归一化至[0, 1]
    ])
    #读取指定目录下图片并转换为Tensor
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('RGB')
            img = transform(img)  # 应用定义的转换器
            img_list.append(img)
    return img_list

class CustomDataset(Dataset):
    def __init__(self, split):
        assert split in ["train", "val"]
        data_root = "violence_224/"
        self.data = [os.path.join(data_root, split, i) for i in os.listdir(data_root + split)]
        if split == "train":
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # 随机翻转
                transforms.ToTensor(),  # 将图像转换为Tensor
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),  # 将图像转换为Tensor
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        x = Image.open(img_path)
        y = int(img_path.split("\\")[-1][0])    #获取标签值，0代表非暴力，1代表暴力
        x = self.transforms(x)
        return x, y

class TensorDataset(Dataset):
    def __init__(self, tensor_list):
        self.data = tensor_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tensor = self.data[index]
        #验证时标签未知，设置默认为 0
        return tensor, 0
class CustomDataModule(LightningDataModule):
    def __init__(self, tensor_lists=None, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        if tensor_lists is None:
            self.tensor_lists = []
        else:
            self.tensor_lists = tensor_lists

    def setup(self, stage=None):

        self.train_dataset = CustomDataset("train")
        self.val_dataset = CustomDataset("val")
        self.test_dataset = TensorDataset(self.tensor_lists)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class ViolenceClassifier(LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
        self.accuracy = Accuracy(task="multiclass", num_classes=2)

        self.list = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # 定义优化器
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        softmax = torch.nn.Softmax()
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        self.list = []
        for i in probabilities:
            if i[0] > i[1]:
                self.list.append(0)
            else:
                self.list.append(1)

        return logits


if __name__ == "__main__":
    #模型训练
    '''
    gpu_id = [1]
    lr = 3e-4
    batch_size = 8
    log_name = "resnet18_pretrain_test"
    print("{} gpu: {}, batch size: {}, lr: {}".format(log_name, gpu_id, batch_size, lr))

    data_module = CustomDataModule(batch_size=batch_size)
    # 设置模型检查点，用于保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=log_name + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    logger = TensorBoardLogger("train_logs", name=log_name)

    # 实例化训练器
    trainer = Trainer(
        max_epochs=10,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    # 实例化模型
    model = ViolenceClassifier(learning_rate=lr)
    # 开始训练
    trainer.fit(model, data_module)
    '''

    #模型测试
    folder_path = 'violence_224/test'  # 替换为实际的文件夹路径

    img_list = load_images_from_folder(folder_path)

    gpu_id = [0]
    batch_size = 128

    data_module = CustomDataModule(img_list, batch_size=batch_size)
    ckpt_path = "violence_model.ckpt"

    model = ViolenceClassifier.load_from_checkpoint(ckpt_path)
    trainer = Trainer(default_root_dir='')
    results = trainer.test(model, data_module)
    list = model.list
    print(list)


