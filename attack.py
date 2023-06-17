import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torchvision.datasets import *
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *
from torchvision.utils import save_image
from tqdm import tqdm
from torchplus.utils import Init, ClassificationAccuracy

if __name__ == "__main__":
    batch_size = 8
    log_epoch = 2
    class_num = 40
    root_dir = "D:/log/paper1/logZZPMAIN.attack"
    dataset_dir = "E:/datasets/at&t face database"
    target_pkl = "D:/log/paper1/logZZPMAIN/Model_Nov25_14-45-14_zzp-asus_main AT and T face/mynet_50.pkl"
    h = 112
    w = 92
    alpha = 50000
    beta = 1000
    gama = 0.001
    learning_rate = 0.1
    momentum = 0.9

    init = Init(
        seed=9970,
        log_root_dir=root_dir,
        backup_filename=__file__,
        tensorboard=True,
        comment=f"attack AT and T face",
    )
    output_device = init.get_device()
    writer = init.get_writer()
    log_dir = init.get_log_dir()
    data_workers = 2

    def Process(im_flatten):
        maxValue = torch.max(im_flatten)
        minValue = torch.min(im_flatten)
        im_flatten = im_flatten - minValue
        im_flatten = im_flatten / (maxValue - minValue)
        return im_flatten

    def Attack(mynet, target_label):
        aim_flatten = torch.zeros(1, h * w).to(output_device)
        v = torch.zeros(1, h * w).to(output_device)
        aim_flatten.requires_grad = True
        costn_1 = 10
        b = 0
        # g = 0
        out = mynet.forward(aim_flatten.detach())
        after_softmax = F.softmax(out, dim=-1)
        predict = torch.argmax(after_softmax)
        writer.add_image(
            f"original input image {target_label}",
            aim_flatten.detach().reshape(1, h, w),
            target_label,
        )
        writer.add_text(
            f"original input image predict label {target_label}",
            f"predict label: {predict.item()}",
        )
        for i in tqdm(range(alpha), desc="Computing"):
            out = mynet.forward(aim_flatten)
            if aim_flatten.grad is not None:
                aim_flatten.grad.zero_()
            out = out.reshape(1, class_num)
            target_class = torch.tensor([target_label]).to(output_device)
            cost = nn.CrossEntropyLoss()(out, target_class)
            cost.backward()
            aim_grad = aim_flatten.grad
            # see https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
            aim_flatten = aim_flatten - learning_rate * (momentum * v + aim_grad)
            aim_flatten = Process(aim_flatten)
            aim_flatten = torch.clamp(aim_flatten.detach(), 0, 1)
            aim_flatten.requires_grad = True
            if cost >= costn_1:
                b = b + 1
                if b > beta:
                    break
            else:
                b = 0
            costn_1 = cost
            if cost < gama:
                break
        out = mynet.forward(aim_flatten.detach())
        after_softmax = F.softmax(out, dim=-1)
        predict = torch.argmax(after_softmax)
        writer.add_image(
            f"inverted image {target_label}",
            aim_flatten.detach().reshape(1, h, w),
            target_label,
        )
        writer.add_text(
            f"inverted image predict label {target_label}",
            f"predict label: {predict.item()}",
        )
        save_image(
            aim_flatten.detach().reshape(h, w), f"{log_dir}/inverted_{target_label}.png"
        )

    class Net(nn.Module):
        def __init__(self, input_features, output_features):
            super(Net, self).__init__()
            self.input_features = input_features
            self.output_features = output_features
            self.regression = nn.Linear(
                in_features=self.input_features, out_features=self.output_features
            )

        def forward(self, x):
            x = self.regression(x)
            return x

    class MLP(nn.Module):
        def __init__(self, input_features, output_features):
            super(MLP, self).__init__()
            self.input_features = input_features
            self.middle_features = 3000
            self.output_features = output_features
            self.fc = nn.Linear(
                in_features=self.input_features, out_features=self.middle_features
            )
            self.regression = nn.Linear(
                in_features=self.middle_features, out_features=self.output_features
            )

        def forward(self, x):
            x = self.fc(x)
            x = self.regression(x)
            return x

    mynet = Net(h * w, class_num).to(output_device).train(False)
    assert os.path.exists(target_pkl)
    mynet.load_state_dict(
        torch.load(open(target_pkl, "rb"), map_location=output_device)
    )
    for i in tqdm(range(class_num), desc="Attack Class"):
        Attack(mynet=mynet, target_label=i)
    writer.close()
