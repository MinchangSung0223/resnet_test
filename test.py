
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import cv2
batch_size = 100
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dataset = dset.ImageFolder(root="test/",transform=transforms.Compose([
                               transforms.Scale(128),       # 한 축을 128로 조절하고
                               transforms.CenterCrop(128),  # square를 한 후,
                               transforms.ToTensor(),       # Tensor로 바꾸고 (0~1로 자동으로 normalize)
                               transforms.Normalize((0.5, 0.5, 0.5),  # -1 ~ 1 사이로 normalize
                                                    (0.5, 0.5, 0.5)), # (c - m)/s 니까...
                           ]))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(1024, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        #print("1:",x.shape)
        out = self.conv(x)
        #print("2:",out.shape)
        out = self.bn(out)
        #print("3:",out.shape)
        out = self.relu(out)
        #print("4:",out.shape)
        out = self.layer1(out)
        #print("5:",out.shape)
        out = self.layer2(out)
        #print("6:",out.shape)
        out = self.layer3(out)
        #print("7:",out.shape)
        out = self.avg_pool(out)
        #print("8:",out.shape)
        out = out.view(out.size(0), -1)
        #print("9:",out.shape)
        out = self.fc(out)
        #print("10:",out.shape)
        return out
model =ResNet(ResidualBlock, [2, 2, 2]).to(device)

model.load_state_dict(torch.load('resnet.ckpt'))
model.eval()
import cv2
import numpy as np
import time
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images_arr = images.detach().cpu().numpy()
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        for i in range(images_arr.shape[0]):
            image = images_arr[i,:,:].transpose((2,1,0))
            image = np.uint8((image*[0.5 ,0.5 ,0.5]+[0.5,0.5,0.5])*255)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            cv2.imshow('test_img',image)
            cv2.waitKey(1)
            time.sleep(1)


        _, predicted = torch.max(outputs.data, 1)
        print("predict : ", predicted )
        print("labels : ",labels)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')
