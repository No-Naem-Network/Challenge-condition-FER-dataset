import argparse
import os,sys,shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
#import transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import math
#from ResNet_MN_Val_all import resnet18, resnet50, resnet101
from part_attention import resnet18, resnet34, resnet50, resnet101
from val_part_attention_sample import MsCelebDataset, CaffeCrop
import scipy.io as sio  
import numpy as np
import pdb
import torch._utils
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--img_dir_val', metavar='DIR', default='/media/sdc/kwang/ferplus/different_pose_ferplus/val/', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='./data/resnet18/checkpoint_40.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--model_dir','-m', default='./model', type=str)
parser.add_argument('--end2end', default=True,\
        help='if true, using end2end with dream block, else, using naive architecture')





def get_val_data(img_name, label, frame_num):
    img_dir_val = '/data/ngocnkd/ngocnkd/region-attention-network/New_Data/FER_valid'
    caffe_crop = CaffeCrop('test')
    # txt_path = '/home/oem/project/Face Expression/5. Challenge-condition-FER-dataset/Data/FER2013Valid/'
    # val_list_file = txt_path+list_txt
    # val_label_file = txt_path+label_txt
    #pdb.set_trace()
    val_dataset =  MsCelebDataset(img_name, img_dir_val, label,
                transforms.Compose([caffe_crop,transforms.ToTensor()]))
    val_loader = torch.utils.data.DataLoader(
            val_dataset,batch_size=frame_num, shuffle=False,
    num_workers=args.workers, pin_memory=True)

    return val_loader


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = np.array(classes)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def main(arch,resume):
    global args
    args = parser.parse_args()
    arch = arch.split('_')[0]
    model = None
    assert(arch in ['resnet18','resnet34','resnet50','resnet101'])
    if arch == 'resnet18':
        model = resnet18(end2end=args.end2end)
    if arch == 'resnet34':
        model = resnet34(end2end=args.end2end)
    if arch == 'resnet50':
        model = resnet50(nverts=nverts,faces=faces,shapeMU=shapeMU,shapePC=shapePC, num_classes=class_num, end2end=args.end2end)
    if arch == 'resnet101':
        model = resnet101(pretrained=False, num_classes=class_num,\
                extract_feature=True, end2end=end2end)


    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    assert(os.path.isfile(resume))
    #pdb.set_trace()

    checkpoint = torch.load(resume)
    #pdb.set_trace()
    # model.load_state_dict(checkpoint['state_dict'])
    # checkpoint = torch.load(resume, map_location='cpu')
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = model.state_dict()
    
    for key in pretrained_state_dict:
        # if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
        #     pass
        # else:    
            model_state_dict[key] = pretrained_state_dict[key]

    model.load_state_dict(model_state_dict, strict = False)

    cudnn.benchmark = True

    # val_nn_txt = '/home/oem/project/Face Expression/5. Challenge-condition-FER-dataset/val_ferplus_mn.txt'
    val_nn_txt = '/data/ngocnkd/ngocnkd/region-attention-network/New_Data/FER_valid/label/ferplus_random_crop_val_list.txt'
    # Fix read without binary mode
    # Old code: val_nn_files = open(val_nn_txt,'rb')
    val_nn_files = open(val_nn_txt,'r')
    correct = 0
    video_num = 0
    output_task1 = open('ferplus_mn_score.txt','w+')
    y_true = []
    y_pred = []
    for val_nn_file in val_nn_files:
        record = val_nn_file.strip().split()
        space_index = val_nn_file.find(' ')
        backslash_index = val_nn_file.find('/')
        
        frame_num = val_nn_file[space_index+1:]
        video_num = video_num +1
        video_name = val_nn_file[backslash_index+1:space_index]
        img_count = val_nn_file[space_index+1:]
        label = val_nn_file[0]
        y_true.append(int(label))
        print('video_name',video_name)

        val_loader = get_val_data(video_name, label, int(frame_num))
        for i,(input,label) in enumerate(val_loader):
            # label = label.numpy()
            input_var = torch.autograd.Variable(input, volatile=True)
            #pdb.set_trace()
            #output, f_need_fix, feature_standard = model(input_var)
            output = model(input_var)
            output_write = output
            output_write =output_write[0]
            output_write = output_write.cpu().data.numpy()
            # print('output_write',output_write)
            #pdb.set_trace()
            output_of_softmax = F.softmax(output,dim=1)
            output_of_softmax_ = output_of_softmax.cpu().data.numpy()
            print('output softmax', output_of_softmax_)
            pred_class = np.argmax(output_of_softmax_)
            #output_of_softmax_ = output_of_softmax_[0]
            #output_task1.write(video_name+' '+str(output_of_softmax_[0])+' '+str(output_of_softmax_[1])+' '+str(output_of_softmax_[2])+' '+str(output_of_softmax_[3])+' '+str(output_of_softmax_[4])+' '+str(output_of_softmax_[5])+' '+str(output_of_softmax_[6])+'\n')
            output_task1.write(video_name+' '+str(pred_class)+'\n')
            pred_final = output_of_softmax[0].data.max(0,keepdim=True)[1]

            #pdb.set_trace()
            #pred_final = pred_final.cpu().data.numpy()
            pred_final = pred_final.cpu().numpy()
            print('True label: ',label[0])
            print('Predict label: ', int(pred_final[0]))
            if int(label[0]) == int(pred_final[0]):
               correct = correct +1
            #    print('predict right label',label[0])
        y_pred.append(int(pred_final[0]))

    print('accuracy', float(correct)/video_num)
    print('correct',correct)
    print('video_num',video_num)
    plot_confusion_matrix(y_true, y_pred, classes=[0, 1, 2, 3, 4, 5, 6, 7])
    fig = plt.gcf()
    plt.show()
    fig.savefig('fig2.png')

if __name__ == '__main__':
    
    #infos = [ ('resnet18_naive', './model/checkpoint_6_654.pth.tar'), 
               #]
	
    # infos = [ ('resnet18_naive', '/media/sdc/kwang/ferplus/pose_test/model_best.pth.tar'), 
    #            ]
    # infos = [ ('resnet18_naive', '/data/ngocnkd/ngocnkd/region-attention-network/model_dir/checkpoint_200.pth.tar'), ]
    infos = [ ('resnet18_naive', '/data/ngocnkd/ngocnkd/region-attention-network/pre_trained_model/Resnet18_FER+_pytorch.pth.tar'), ]
    # infos = [ ('resnet18_naive', '/data/ngocnkd/ngocnkd/region-attention-network/pre_trained_model/Resnet18_MS1M_pytorch.pth.tar'), ]


    for arch, model_path in infos:
        print("{} {}".format(arch, model_path))
        main(arch, model_path)
        
        print()