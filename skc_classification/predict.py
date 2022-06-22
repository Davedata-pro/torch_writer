import collections
import argparse
import torch
from torch import nn
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader
import os
import shutil
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import time
from PIL import Image

#with torch.no_grad():
#    model.eval()

### 1개 테스트
## 1. image path 및 predict list
#img_path = r"C:\Users\22005604\Desktop\20220608_khj\0614_RGB_oriGbrBin\val\0"
#img_data = os.listdir(img_path)


### 2. PIL open
#img_origin = Image.open(img_path)

### 3. transform 형식
#mean_train = [0.485, 0.456, 0.406]
#std_train = [0.229, 0.224, 0.225]

#transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(232), transforms.CenterCrop(224), 
#                                transforms.Normalize(mean = mean_train, std= std_train)])

#weight_path = r"C:\Users\22005604\Desktop\20220608_khj\weight\0615_oriGbrBin_5class\0616_origin_2drop(2)\jit_L0.137092_A0.936_71.pt"

### 4. model load 및 transform(img) = img
#model = torch.jit.load(weight_path)
#img = transform(img_origin)

### 5. img = torch.unsqueeze(img, 0).to('cuda') (차원 맞추고, cuda 지정)
#img = torch.unsqueeze(img, 0).to("cuda")
#result = model(img)
#_, preds = torch.max(result, 1)
#print("results:", result)
#print("preds:", preds)

### 1개 테스트

######### 폴더별 테스트 @#############

## 1. image path 및 predict list
val_folder = "test2"
img_path = r"C:\Users\22005604\Desktop\20220615_pdh\0614_RGB_oriGbrBin\val\{}".format(val_folder)
img_data = os.listdir(img_path)

if str(val_folder) not in os.listdir("./predict"):
    os.mkdir("./predict/"+"{}".format(val_folder))

weight_path = r"C:\Users\22005604\Desktop\20220615_pdh\weight\jit_L0.000000_A0.944_113.pt"
model = torch.jit.load(weight_path)

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

### 2. PIL open
for image in img_data:
    img_origin = Image.open(img_path+"/"+image)
    print(type(img_origin))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(232), transforms.CenterCrop(224), 
                                transforms.Normalize(mean = mean_train, std= std_train)])
## 3. transform 형식
    img = transform(img_origin)
    img = torch.unsqueeze(img, 0).to("cuda")
    result = model(img)
    top, preds = torch.max(result, 1)
    # print(result)
    results_list = torch.sort(result, dim=1, descending=True)
    print(results_list)
    line = np.sort(result.cpu().detach().numpy())
    top2 = line.flatten()[::-1][:2]
    print("상위 결과 2개만:", top2) # 소장님 softmax 값 확인
    print("\n")
    
    str_preds = str(preds)
    # print(type(top2[0]))
    class_number = str_preds[8:9]
    #import pdb;pdb.set_trace()

    if class_number == "0":
        class_number = "Short"
    elif class_number == "1":
        class_number = "Size"
    elif class_number == "2":
        class_number = "Surface"
    elif class_number == "3":
        class_number = "Debris"
    elif class_number == "4":
        class_number = "Hole"

    image_name = image.split(".b")[0]
    img_origin.save('./predict/{}/{}_{}_{:.2f}.bmp'.format(val_folder, image_name, class_number, top2[0]))