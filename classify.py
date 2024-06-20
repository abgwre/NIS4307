import torch
from pytorch_lightning import Trainer

from _13_other_files_directory.model import ViolenceClassifier, CustomDataModule, load_images_from_folder

class ViolenceClass:
    def __init__(self):
        # 加载模型、设置参数等
        ckpt_path = "violence_model.ckpt"
        self.model = ViolenceClassifier.load_from_checkpoint(ckpt_path)
        self.trainer = Trainer()

    def misc(self):
        # 其他处理函数
        ...

    def classify(self, imgs: torch.Tensor) -> list:
        # 图像分类
        data_module = CustomDataModule(imgs, batch_size=128)
        results = self.trainer.test(self.model, data_module)
        preds = self.model.list
        return preds


