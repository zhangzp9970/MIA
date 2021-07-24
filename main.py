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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import *
from tqdm import tqdm

gpus = 0
data_workers = 0
batch_size = 64
min_step = 100*150
log_interval = 100
test_interval = 100
classes = 40
root_dir = "./log"
train_file = "C:\\Users\\zhang\\attfdbtrain.txt"
test_file = "C:\\Users\\zhang\\attfdbtest.txt"

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

train_transform = Compose([
    Grayscale(num_output_channels=1),
    ToTensor()
])

test_transform = Compose([
    Grayscale(num_output_channels=1),
    ToTensor()
])

train_ds = FileListDataset(list_path=train_file, transform=train_transform)
test_ds = FileListDataset(list_path=test_file, transform=test_transform)

train_dl = DataLoader(dataset=train_ds, batch_size=batch_size,
                      shuffle=True, num_workers=data_workers, drop_last=True)
test_dl = DataLoader(dataset=test_ds, batch_size=batch_size,
                     shuffle=False, num_workers=data_workers, drop_last=False)


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
                        output_device=output_device).train(True)
optimizer = optim.SGD(mynet.parameters(), lr=0.01)

global_step = 0
best_acc = 0
total_steps = tqdm(range(min_step), desc='global step')
epoch_id = 0

while global_step < min_step:
    iters = tqdm(train_dl, desc=f'epoch {epoch_id} ', total=len(train_dl))
    epoch_id += 1
    for i, (im, label) in enumerate(iters):
        im = im.to(output_device)
        label = label.to(output_device)
        bs = im.shape[0]
        im_flatten = im.reshape([bs, -1])
        out = mynet.forward(im_flatten)
        ce = nn.CrossEntropyLoss()(out, label)
        with OptimizerManager([optimizer]):
            loss = ce
            loss.backward()

        global_step += 1
        total_steps.update()
        if global_step % log_interval == 0:
            counter = AccuracyCounter()
            counter.addOneBatch(variable_to_numpy(
                one_hot(label, classes)), variable_to_numpy(F.softmax(out, dim=-1)))
            acc_train = torch.tensor(
                [counter.reportAccuracy()]).to(output_device)
            logger.add_scalar('loss', loss, global_step)
            logger.add_scalar('acc_train', acc_train, global_step)

        if global_step % test_interval == 0:
            counters = [AccuracyCounter() for x in range(classes)]
            with TrainingModeManager([mynet], train=False) as mgr, \
                    Accumulator(['after_softmax', 'label']) as target_accumulator, \
                    torch.no_grad():
                for i, (im, label) in enumerate(tqdm(test_dl, desc='testing ')):
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs = im.shape[0]
                    im_flatten = im.reshape([bs, -1])
                    out = mynet.forward(im_flatten)
                    after_softmax = F.softmax(out, dim=-1)

                    for name in target_accumulator.names:
                        globals()[name] = variable_to_numpy(globals()[name])

                    target_accumulator.updateData(globals())

            for x in target_accumulator:
                globals()[x] = target_accumulator[x]

            counters = [AccuracyCounter() for x in range(classes)]
            for (each_predict_prob, each_label) in zip(after_softmax, label):
                counters[each_label].Ntotal += 1.0
                each_pred_id = np.argmax(each_predict_prob)
                if each_pred_id == each_label:
                    counters[each_label].Ncorrect += 1.0

            acc_tests = [x.reportAccuracy()
                         for x in counters if not np.isnan(x.reportAccuracy())]
            acc_test = torch.ones(1, 1) * np.mean(acc_tests)
            logger.add_scalar('acc_test', acc_test, global_step)
            clear_output()
            data = {
                "mynet": mynet.state_dict(),
            }
            if acc_test > best_acc:
                best_acc = acc_test
                with open(os.path.join(log_dir, 'best.pkl'), 'wb') as f:
                    torch.save(data, f)

                with open(os.path.join(log_dir, 'current.pkl'), 'wb') as f:
                    torch.save(data, f)
logger.close()
