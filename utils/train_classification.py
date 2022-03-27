from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import PhaseDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


#########ADDING ARGUMENTS##########

parser = argparse.ArgumentParser()
parser.add_argument(
    '-b','--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '-n','--num_points', type=int, default=2000, help='input batch size')
parser.add_argument(
    '-w','--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '-e','--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('-of','--outf', type=str, default='cls', help='output folder')
parser.add_argument('-m','--model', type=str, default='', help='model path')
parser.add_argument('-d','--dataset', type=str, required=True, help="dataset path")
parser.add_argument('-f','--feature_transform', action='store_true', help="use feature transform")
opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[91m' + x + '\033[0m'

#########SEED EVERYTHING##########

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

#########CHOOSING DATASET##########

dataset = PhaseDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

test_dataset = PhaseDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
    
#########LOADING TRAIN & TEST DATA##########

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

#########SETTING UP CLASSIFIER##########
classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()
num_batch = len(dataset) / opt.batchSize

#output test loss and accuracy
test_out = ('test_out_npts_%d.dat' %(opt.num_points))
ftest = open(test_out,'w')
ftest.write('num_epoch loss %s acc_overall\n' % 
            (' '.join('class'+str(num) for num in range(num_classes))))
train_out = ('train_out_npts_%d.dat' %(opt.num_points))
ftrain = open(train_out,'w')
ftrain.write('num_epoch loss %s acc_overall\n' % 
             (' '.join('class' + str(num) for num in range(num_classes))))

#########TRAINING THE MODEL##########
model_path = '%s/pts_%s' % (opt.outf, str(opt.num_points))
if not os.path.exists(model_path):
    os.mkdir(model_path)

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        
        # write out the train loss and accuracy
        acc_class = []
        for m in range(num_classes):
            ind_m = (target.data.cpu().numpy().ravel() == m)           
            if np.sum(ind_m) == 0:
                acc_m = 0
            else:
                acc_m = np.sum((pred_choice.data.cpu().numpy().ravel()[ind_m] == m))/np.sum(ind_m)
            acc_class.append(acc_m)
        
        print(f"[{epoch}: {i}/{int(num_batch)}] train loss: {loss.item()} accuracy: {correct.item()/float(opt.batchSize):.3f}")
        ftrain.write(f"{epoch} {loss.item()} {correct.item() / float(opt.batchSize)} \n")
        
        if i % (num_batch+1) == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            
            # write out the test loss and accuracy
            acc_class = []
            for m in range(num_classes):
                ind_m = (target.data.cpu().numpy().ravel() == m)           
                if np.sum(ind_m) == 0:
                    acc_m = 0
                else:
                    acc_m = np.sum((pred_choice.data.cpu().numpy().ravel()[ind_m] == m))/np.sum(ind_m)
                acc_class.append(acc_m)

            print(f"[{epoch}: {i}/{int(num_batch)}] {blue('test')} loss: {loss.item()} accuracy: {correct.item()/float(opt.batchSize):.3f}")
            ftest.write(f"{epoch} {loss.item()} {correct.item() / float(opt.batchSize)} \n")
        
    torch.save(classifier.state_dict(), '%s/pts_%s/cls_model_%d.pth' % (opt.outf, str(opt.num_points), epoch))

# Close the logs for train and test
ftest.close()
ftrain.close()

#########TESTING##########

# Initialization for testing the trained model
total_correct = 0
total_testset = 0
acc_classes = []
conf_mats = []

for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    
    acc_class = []
    conf_mat = np.zeros((num_classes, num_classes))
    for n in range(num_classes):
        for m in range(num_classes):
            ind_m = (target.data.cpu().numpy().ravel() == m) 
            if np.sum(ind_m) != 0:
                conf_mat[m, n] = np.sum((pred_choice.data.cpu().numpy().ravel()[ind_m] == n)) 
                
            # Record the diagonal elements to calculate accuracy
            if m == n: 
                if np.sum(ind_m) == 0:
                    acc_m = 0
                else:
                    acc_m = np.sum((pred_choice.data.cpu().numpy().ravel()[ind_m] == m)) / np.sum(ind_m)
                acc_class.append(acc_m)            
    
    acc_classes.append(acc_class)
    conf_mats.append(conf_mat)
    total_correct += correct.item()
    total_testset += points.size()[0]
      
final_acc = total_correct / float(total_testset)

ave_conf_mats = np.mean(conf_mats,0)
print("Final accuracy {}".format(final_acc))


# Save the confusion matrix for computing accuracy, precisions, recalls, and f1 scores.
np.savetxt(("conf_mat_%d.dat" % opt.num_points), np.sum(conf_mats, 0), fmt='% 4d', delimiter=',')
