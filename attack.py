import datetime
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from easydl import *
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import *
from tqdm import tqdm

gpus = 0
data_workers = 0
# batch_size = 64
classes = 40
root_dir = "./log"
test_pkl = "C:\\Users\\zhang\\Documents\\GitHub\\MIA\\log\\Jul24_14-54-36\\best.pkl"

alpha = 50000
beta = 1000
gama = 0.001
learning_rate = 0.1
momentum = 0.9

cudnn.benchmark = True
cudnn.deterministic = True
seed = 9970
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

if gpus < 1:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpu_ids = []
    output_device = torch.device('cpu')
else:
    gpu_ids = select_GPUs(gpus)
    output_device = gpu_ids[0]

now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = f'{root_dir}/{now}'
logger = SummaryWriter(log_dir)


def Process(im_flatten):
    maxValue = torch.max(im_flatten)
    minValue = torch.min(im_flatten)
    im_flatten = im_flatten-minValue
    im_flatten = im_flatten/(maxValue-minValue)
    return im_flatten


def Attack(mynet, target_label):
    aim_flatten = torch.zeros(1, 112*92)
    v = torch.zeros(1, 112*92)
    aim_flatten.requires_grad = True
    costn_1 = 10
    b = 0
    g = 0
    out = mynet.forward(aim_flatten.detach())
    after_softmax = F.softmax(out, dim=-1)
    predict = torch.argmax(after_softmax)
    print(predict)
    logger.add_image(f'original input image {target_label}',
                     aim_flatten.detach().reshape(1, 112, 92), target_label)
    logger.add_text(f'original input image predict label {target_label}',
                    f'predict label: {predict.item()}')
    for i in range(alpha):
        out = mynet.forward(aim_flatten)
        if aim_flatten.grad is not None:
            aim_flatten.grad.zero_()
        out = out.reshape(1, 40)
        target_class = torch.tensor([target_label])
        cost = nn.CrossEntropyLoss()(out, target_class)
        cost.backward()
        aim_grad = aim_flatten.grad
        # see https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        aim_flatten = aim_flatten-learning_rate*(momentum*v+aim_grad)
        aim_flatten = Process(aim_flatten)
        aim_flatten = torch.clamp(aim_flatten.detach(), 0, 1)
        aim_flatten.requires_grad = True
        if cost >= costn_1:
            b = b+1
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
    print(predict)
    logger.add_image(f'inverted image {target_label}',
                     aim_flatten.detach().reshape(1, 112, 92), target_label)
    logger.add_text(f'inverted image predict label {target_label}',
                    f'predict label: {predict.item()}')
    oim_flatten = aim_flatten.detach()
    output_image = oim_flatten.reshape(112, 92)
    output_image = output_image*255
    output_imageN = output_image.numpy()
    im = Image.fromarray(np.uint8(output_imageN))
    im.save(f'{log_dir}/{target_label}.pgm')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input_features = 112*92
        self.output_features = classes
        self.regression = nn.Linear(
            in_features=self.input_features, out_features=self.output_features)

    def forward(self, x):
        x = self.regression(x)
        return x


net = Net()
mynet = nn.DataParallel(net, device_ids=gpu_ids,
                        output_device=output_device).train(False)
assert os.path.exists(test_pkl)
data = torch.load(open(test_pkl, 'rb'))
mynet.load_state_dict(data['mynet'])
for i in range(40):
    clear_output()
    print(f'---class{i}---')
    Attack(mynet=mynet, target_label=i)
logger.close()
