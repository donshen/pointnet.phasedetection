from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ShapeNetDataset
from pointnet.model_stn import PointNetCls
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', type=str, default = '',  help='model path')
parser.add_argument('-n', '--num_points', type=int, default=2000, help='input batch size')


opt = parser.parse_args()
print(opt)

test_dataset = ShapeNetDataset(
    root='/mnt/ssd1/donny/test_pn/pointnet.pytorch/point_clouds',
    split='test',
    classification=True,
    npoints=opt.num_points,
    data_augmentation=False)

testdataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=True)

num_classes = len(test_dataset.classes)
classifier = PointNetCls(k=num_classes, feature_transform=True)
#classifier = PointNetCls(k=len(test_dataset.classes))
classifier.cuda()
#classifier.load_state_dict(torch.load(opt.model))
#classifier.eval()
classifier.load_state_dict(torch.load(opt.model), strict=False)


pred_class = []

for i, data in enumerate(testdataloader, 0):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    loss = F.nll_loss(pred, target)

    pred_choice = pred.data.max(1)[1]
    
    
    
    #correct = pred_choice.eq(target.data).cpu().sum()
    correct = pred_choice.eq(target.data).cpu().sum().data.numpy()
    pred_class = np.concatenate((pred_class, pred_choice.data.cpu().numpy().ravel()), axis=None)
    print('i:%d  loss: %f accuracy: %f' % (i, loss.data.item(), correct / float(32)))

vals = [0 for _ in range(9)]
for cls in pred_class:
    vals[int(cls)] += 1
bins = [i for i in range(9)]
plt.bar(bins, vals)
plt.savefig('pred_hist.png', dpi = 100)
