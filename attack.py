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
test_pkl = "D:\\best.pkl"
# attack_image_path = "D:\\attfdbtest\\3\\9.pgm"
target_label = 12

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


attack_transform = Compose([
    Grayscale(num_output_channels=1),
    ToTensor()
])


# def LoadAttackImage(image_path):
#     assert os.path.exists(image_path)
#     im = Image.open(image_path).convert('RGB')
#     im = attack_transform(im)
#     return im


def Process(im_flatten):
    maxValue = torch.max(im_flatten)
    minValue = torch.min(im_flatten)
    im_flatten = im_flatten-minValue
    im_flatten = im_flatten/(maxValue-minValue)
    return im_flatten


def Cost(NNout, label):
    ithNNout = NNout[label]
    cost = 1.0-ithNNout
    return cost


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.regression = nn.Linear(in_features=92*112, out_features=classes)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.regression(x)
        # x = self.softmax(x)
        return x


net = Net()
mynet = nn.DataParallel(net, device_ids=gpu_ids,
                        output_device=output_device).train(False)
assert os.path.exists(test_pkl)
data = torch.load(open(test_pkl, 'rb'))
mynet.load_state_dict(data['mynet'])

# attack_image = LoadAttackImage(attack_image_path)
# aim_shape = attack_image.shape
# attack_image = attack_image.to(output_device)
# aim_flatten = attack_image.flatten()
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


for i in range(alpha):

    out = mynet.forward(aim_flatten)
    # after_softmax = F.softmax(out, dim=-1)

    if aim_flatten.grad is not None:
        aim_flatten.grad.zero_()

    # cost = Cost(after_softmax, target_class)
    out = out.reshape(1, 40)
    target_class = torch.tensor([target_label])
    cost = nn.CrossEntropyLoss()(out, target_class)

    cost.backward()
    print(cost)

    aim_grad = aim_flatten.grad
    # see https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
    aim_flatten = aim_flatten-learning_rate*(momentum*v+aim_grad)
    aim_flatten = Process(aim_flatten)
    aim_flatten = torch.clamp(aim_flatten.detach(), 0, 1)
    aim_flatten.requires_grad = True

    if cost >= costn_1:
        b = b+1
        if b > beta:
            print("break here b")
            break
    else:
        b = 0

    costn_1 = cost
    if cost < gama:
        break
    #     g = g+1
    #     print(g)
    #     if g > 1:
    #         print("break here g")
    #         break
    # else:
    #     g = 0

    if (i+1) % 10 == 0:
        # print(cost)
        logger.add_scalar('cost', cost, i)

out = mynet.forward(aim_flatten.detach())
after_softmax = F.softmax(out, dim=-1)
predict = torch.argmax(after_softmax)
print(predict)

oim_flatten = aim_flatten.detach()
output_image = oim_flatten.reshape(112, 92)
output_image = output_image*255
output_imageN = output_image.numpy()
im = Image.fromarray(np.uint8(output_imageN))
im.save(log_dir+"/"+"hhhh.pgm")
im.show()
logger.add_scalar('cost', cost, i)
