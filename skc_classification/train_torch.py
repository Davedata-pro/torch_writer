import collections
import argparse
import torch
from torch import nn
import numpy as np
import torchvision
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader
import os
import shutil
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import time
# from torch.utils.tensorboard import SummaryWriter


## initial weight 파일 받아오는 부분
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


## image net 에서 weight 가져오는데, 전체 이미지들의 평균과 분산을 지정해줌
## pixel value 표준화
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

cfgs = {  # 모델 시리얼 넘버
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'PAD': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'], # vgg16 modified(for HDL) : conv1 ~ 4, remove conv5
    #'PAD': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M'], # vgg16 modified(for HDL) : conv1 ~ 4, remove conv5
    #'PAD': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M'], # vgg16 modified(for HDL) : conv1 ~ 4, remove conv5
    'HDL': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'], # vgg16 modified(for HDL) : conv1 ~ 4, remove conv5
    #'HDL': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'], # vgg16 modified(for HDL) : conv1 ~ 4, remove conv5
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


# torchvision.transforms : 데이터를 불러오면서 바로 전처리

def data_transforms(load_size=232, input_size=224):  ## 256으로 리사이즈하고, 224로 random crop
    data_transforms = {
        'train' : transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(load_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),   # interpolation 수정 
            #transforms.Resize(load_size),   # interpolation 수정 
            #transforms.RandomResizedCrop(input_size),
            transforms.RandomCrop(size=(input_size, input_size)),
            #transforms.RandomRotation(),
            # transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=mean_train,
                                std=std_train)]),
        'val' : transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(load_size),
            transforms.CenterCrop(input_size),
            transforms.Normalize(mean=mean_train,
                                std=std_train)])}
    return data_transforms


# 현재 사용하지 않음
def data_transforms_inv():
    data_transforms_inv = transforms.Compose([transforms.Normalize(mean=list(-np.divide(mean_train, std_train)), std=list(np.divide(1, std_train)))])
    return data_transforms_inv


# 현재 사용하지 않음 (코드나 내용 수정시, 코드와 결과가 함께 나오게 기록하는 용으로? 사용하신듯)
def copy_files(src, dst, ignores=[]):
    src_files = os.listdir(src)
    for file_name in src_files:
        ignore_check = [True for i in ignores if i in file_name]
        if ignore_check:
            continue
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dst, file_name))
        if os.path.isdir(full_file_name):
            os.makedirs(os.path.join(dst, file_name), exist_ok=True)
            copy_files(full_file_name, os.path.join(dst, file_name), ignores)


def get_args():
    parser = argparse.ArgumentParser(description='CLASSIFICATION')
    parser.save_weight = True
    parser.add_argument('--dataset_path', default=r'C:\Users\22005604\Desktop\20220615_pdh\0614_RGB_oriGbrBin')  # 인풋
    parser.add_argument('--num_epoch', default=100) # 보통 200
    parser.add_argument('--lr', default=0.0005)  # 실험 필요
    parser.add_argument('--batch_size', default=8) 
    parser.add_argument('--save_false_img', default=True)
    parser.add_argument('--save_weight', default=True)
    parser.add_argument('--num_class', default=5)
    parser.add_argument('--load_size', default=224) ## 232
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--use_batchnorm', default=False)
    args = parser.parse_args()
    return args

class VGG(nn.Module):
    def __init__(self, features, num_classes, init_weights=True):
        super(VGG, self).__init__()
        self.features = features  # size : (14, 14) #?
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # original : (7,7)
        self.classifier = nn.Sequential(
            nn.Dropout(), ## 지정 필요
            # nn.Linear(512, num_classes),                # for HDL
            nn.Linear(256, num_classes),                # for PAD, origin 256
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(1024, num_classes),
        # )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16(pretrained, arch, cfg, num_class, batch_norm=False, progress=True,  **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), num_classes=num_class, **kwargs)
    model_dict = model.state_dict()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        # filter out unnecessary keys & update weights
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and "classifier" not in k}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    return model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    args = get_args()
    dataset_path = args.dataset_path
    num_epochs = args.num_epoch
    lr = args.lr
    batch_size = args.batch_size
    save_false_img = args.save_false_img
    save_weight = args.save_weight
    num_class = args.num_class
    load_size = args.load_size
    input_size = args.input_size
    use_batchnorm = args.use_batchnorm
    arch = 'vgg16_bn' if use_batchnorm else 'vgg16'

    ##########################################
    ###         Define Network             ###
    ##########################################
    model = vgg16(arch=arch, cfg='PAD', num_class=num_class, pretrained=True, batch_norm=use_batchnorm).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # torch.save(model, "weight/1111.pt")
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save('weight/model_scripted.pt') # Save
    
    # import pdb;pdb.set_trace()
    ##########################################
    ###         Define Dataset             ###
    ##########################################
    data_transforms = data_transforms()
    data_transforms_inv = data_transforms_inv()
    image_datasets = {x: datasets.ImageFolder(root=os.path.join(dataset_path, x),
                                              transform=data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=0) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    class_dict = {v: k for k, v in
                  image_datasets['train'].class_to_idx.items()}  # invert "class_to_idx" => {label:'classname',...}

    # writer = SummaryWriter('summary')
    start_time = time.time()
    global_step = 0
    max_acc = 0.0
    min_loss = 10000
    cnt = 0
    for epoch in range(num_epochs):
        print('-' * 20)
        print('Time consumed: {}s'.format(time.time() - start_time))
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        # print('Dataset size : Train set - {}, Validation set - {}'.format(dataset_sizes['train'], dataset_sizes['validation']))
        print('-' * 20)

        for phase in ['train', 'val']:  # train -> validation
            if phase == 'train':
                print("Train Phase")
                model.train()
            else:
                aa = time.time()
                print("Validation Phase")
                model.eval()

            loss = 0
            pred_list = []
            label_list = []
            batch_list = []
            for idx, (batch, labels) in enumerate(dataloaders[phase]):  # batch loop
                global_step += 1
                batch = batch.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(batch)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)  # outputs : one-hot , labels : class number
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    if phase == 'val':
                        pred_list.extend(preds.tolist())
                        label_list.extend(labels.tolist())

                if idx % 10 == 0:
                    print('Epoch: {} | Step: {} | Loss: {:.4f}'.format(epoch + 1, idx + 1, float(loss.data)))
                    # writer.add_scalar('loss', scalar_value=float(loss.data), global_step=global_step)

            if phase == 'val':
                cf = confusion_matrix(label_list, pred_list)
                acc = sum(cf[i][i] for i in range(num_class)) / len(label_list)
                print(cf)
                print(classification_report(label_list, pred_list, digits=3))
                print('acc: {}, loss: {}'.format(acc, loss))


                print('tt = {}'.format(time.time() - aa))

                if save_weight:
                    if max_acc <= acc:
                        if acc == 1:
                            cnt += 1
                            max_acc = acc
                            if loss < min_loss:
                                torch.save(model, "weight/1111.pt")

                                # torch.save(model.state_dict(), "weight/vgg_0609_L{:.6f}_A{:.3f}_{}.pt".format(loss, acc, epoch + 1))
                                # 모델의 가중치와 편향들은 모델의 매개변수에 포함되어 있음 (model.parameters()로 접근 가능)
                                # state_dict: 각 계층을 매개변수 텐서로 매핑되는 dict 객체..
                                # 학습 가능한 매개변수를 갖는 계층(합성곱, linear 등) 및 등록된 버퍼(배치놈의 running_mean)들만이 모델의 state_dic에 항목 가짐
                                # 옵티마이저 객체 또한 옵티마이저의 상태뿐만 아니라 사용된 하이퍼 파라미터 정보가 포함된 state_dict 가짐
                                # torch.save(optimizer.state_dict(), "weight/opt_0609_L{:.6f}_A{:.3f}_{}.pt".format(loss, acc, epoch + 1))
                                min_loss = loss
                                #Pytorch에서 모델의 state_dict은 학습가능한 매개변수(가중치, 편항)가 담겨있는 딕셔너리(Dictionary)
                                #매개변수 이외에는 정보가 담겨있지 않기 때문에, 코드 상으로 모델이 구현되어 있는 경우에만
                                #로드하는 방법을 통해 사용할 수 있음

                        else:
                            torch.save(model, "weight/1111.pt")
                            # torch.save(model.state_dict(), "weight/vgg_0609_L{:.6f}_A{:.3f}_{}.pt".format(loss, acc, epoch + 1))
                            # torch.save(optimizer.state_dict(), "weight/opt_0609_L{:.6f}_A{:.3f}_{}.pt".format(loss, acc, epoch + 1))
                            max_acc = acc
                            min_loss = loss

        if (cnt >= 3) and (min_loss < 0.001):
            print('Time : {}'.format(time.time() - start_time))
            break

    print('Time : {}'.format(time.time() - start_time))
