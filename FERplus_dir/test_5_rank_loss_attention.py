import os, sys, shutil
import time

import torch
import torch.backends.cudnn as cudnn
import math
import numpy as np
import scipy.io as sio
from part_attention import resnet18
from val_part_attention_sample import CaffeCrop
import torch.utils
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch._utils
import csv
from PIL import Image

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


class MsCelebDataset(data.Dataset):
    def __init__(self, img_dir, img_info, transform=None):
        self.img_lists = img_info
        self.transform = transform

    def __getitem__(self, index):
        # Read image and tranform (crop image)
        path_fisrt, target_first = self.img_lists[index]
        img_first = Image.open(path_fisrt).convert("RGB")
        if self.transform is not None:
            img_first = self.transform(img_first)
        
        return img_first, target_first 
    
    def __len__(self):
        return len(self.img_lists)

def get_val_data(val_label_content, img_dir):
    # Image infomation
    img_name = val_label_content[1]
    label = val_label_content[2]
    img_path = os.path.join(img_dir, img_name)
    img_info = [(img_path, label)]

    # Crop tranform
    caffe_crop = CaffeCrop('test')

    # Make Dataset and tranform to put in dataloader
    val_dataset =  MsCelebDataset(img_dir, img_info,
                transforms.Compose([caffe_crop,transforms.ToTensor()]))

    # Put data to DataLoader
    val_loader = torch.utils.data.DataLoader(
            val_dataset,batch_size=1, shuffle=False,
    num_workers=16, pin_memory=True)

    return val_loader


def main(arch,resume):
    args_end2end = True
    model = resnet18(end2end=args_end2end)

    # Without Cuda    
    model = torch.nn.DataParallel(model)
    # .cuda()
    model.eval()
    assert(os.path.isfile(resume))

    # Load pretrain model and state dict ----------------------------------
    checkpoint = torch.load(resume, map_location='cpu')
    print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = model.state_dict()
    
    for key in pretrained_state_dict:
        if ((key=='module.fc.weight')|(key=='module.fc.bias')):
            pass
        else:    
            model_state_dict[key] = pretrained_state_dict[key]

    model.load_state_dict(model_state_dict, strict = False)
    cudnn.benchmark = True
    #---------------------------------------------------------------------


    #Img dir
    IMG_DIR = '/home/oem/project/Face Expression/5. Challenge-condition-FER-dataset/Data/FER2013Valid/'
    correct = 0    
    # Load label file
    val_label_csv = '/home/oem/project/Face Expression/5. Challenge-condition-FER-dataset/Data/ferplus_new_val.csv'
    with open(val_label_csv, 'r') as val_label_contents:
        output_task1 = open('ferplus_result.txt', 'w+')
        reader = csv.reader(val_label_contents)
        next(reader, None)  # skip the headers
        # Create data loadder    
        for vlc in reader:
            val_loader = get_val_data(vlc, IMG_DIR)
            for i, (input, label) in enumerate(val_loader):
                print("Label", label)
                # print(input)
                input_var = torch.autograd.Variable(input, volatile=True)
                # pdb.set_trace()
                output, f_need_fix, feature_standard = model(input_var)
                output = model(input_var)
                output_write = output
                output_write =output_write[0]
                output_write = output_write.cpu().data.numpy()
                print('output_write',output_write)
                #pdb.set_trace()
                output_of_softmax = F.softmax(output,dim=1)
                output_of_softmax_ = output_of_softmax.cpu().data.numpy()
                pred_class = np.argmax(output_of_softmax_)
                #output_of_softmax_ = output_of_softmax_[0]
                #output_task1.write(video_name+' '+str(output_of_softmax_[0])+' '+str(output_of_softmax_[1])+' '+str(output_of_softmax_[2])+' '+str(output_of_softmax_[3])+' '+str(output_of_softmax_[4])+' '+str(output_of_softmax_[5])+' '+str(output_of_softmax_[6])+'\n')
                output_task1.write(video_name+' '+str(pred_class)+'\n')
                pred_final = output_of_softmax[0].data.max(0,keepdim=True)[1]
                #pdb.set_trace()
                #pred_final = pred_final.cpu().data.numpy()
                pred_final = pred_final.cpu().numpy()
                if int(label[0]) == int(pred_final[0]):
                    correct = correct +1
                    print('predict right label',label[0])
    print('accuracy', float(correct)/video_num)
    print('correct',correct)
    print('video_num',video_num)






if __name__ == '__main__':
    infos = [ ('resnet18_naive', '/home/oem/project/Face Expression/5. Challenge-condition-FER-dataset/ijba_res18_naive.pth.tar'), 
               ]
    
    for arch, model_path in infos:
        print("{} {}".format(arch, model_path))
        main(arch, model_path)
        
        print()