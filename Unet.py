import numpy as np
from PIL import Image
import cv2
import random
from tqdm import tqdm

def pre_processing(data):

    train_imgs = dataset_normalized(data)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs/255.
    return train_imgs

# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized

def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs
import numpy as np
from PIL import Image
import cv2
import random
from tqdm import tqdm

def pre_processing(data):

    train_imgs = dataset_normalized(data)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs/255.
    return train_imgs

# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized

def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.optim as optim
from torch import nn
import torch.nn.functional as F

def extract_random(full_imgs, full_masks, patch_h, patch_w, N_patches):

    patches = np.empty((N_patches, full_imgs.shape[1], patch_h, patch_w))
    patches_masks = np.empty((N_patches, full_masks.shape[1], patch_h, patch_w))
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches/full_imgs.shape[0])  #N_patches equally divided in the full images
    print("patches per full image: " +str(patch_per_img))
    iter_tot = 0   #iter over the total numbe rof patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        k=0
        while k <patch_per_img:
            x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
            # print "x_center " +str(x_center)
            y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
            # print "y_center " +str(y_center)
            patch = full_imgs[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patch_mask = full_masks[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patches[iter_tot]=patch
            patches_masks[iter_tot]=patch_mask
            iter_tot +=1   #total
            k+=1  #per full_img
    return patches, patches_masks

import torch.utils.data as data

class RetinalDataset(data.Dataset):
    def __init__(self, phase):
        super(RetinalDataset, self).__init__()

        if phase == 'train':
            train_images = np.zeros((6, 1, 584, 565))
            for i in range(6):
                path = 'data/train/'+str(i+1)+'.png'
                train_img = Image.open(path).convert('L')
                train_img = np.array(train_img)
                train_images[i] = train_img
            train_images = pre_processing(train_images)

            gt_images = np.zeros((6, 1, 584, 565))
            for i in range(6):
                path = 'data/gt/'+str(i+1)+'.png'
                gt_img = Image.open(path).convert('1')
                gt_img = np.array(gt_img)
                gt_images[i] = gt_img

            train_images = train_images[:,:,9:574,:]
            gt_images = gt_images[:,:,9:574,:]

            self.patches_imgs, self.patches_masks = \
                        extract_random(train_images, gt_images, 48, 48, 190000)

        else:
            test_img = Image.open('data/train/24.png').convert('L')
            test_input = np.array(test_img)
            test_input = np.expand_dims(test_input, axis=0)
            test_input = np.expand_dims(test_input, axis=0)
            test_input = pre_processing(test_input)
            test_imgs = paint_border(test_input, patch_h=48, patch_w=48)
            self.patches_imgs = extract_ordered(test_imgs, patch_h=48, patch_w=48)


            gt_img = np.array(Image.open('data/gt/24.png').convert('1'))
            gt_img = np.expand_dims(gt_img, axis=0)
            gt_img = np.expand_dims(gt_img, axis=0)
            gt_imgs = paint_border(gt_img, patch_h=48, patch_w=48)
            self.patches_masks = extract_ordered(gt_imgs, patch_h=48, patch_w=48)

    def __getitem__(self, index):

        return self.patches_imgs[index], self.patches_masks[index]

    def __len__(self):
        return self.patches_imgs.shape[0]

def paint_border(data,patch_h,patch_w):
    assert (len(data.shape)==4)  #4D arrays
    assert (data.shape[1]==1 or data.shape[1]==3)  #check the channel is 1 or 3
    img_h=data.shape[2]
    img_w=data.shape[3]
    new_img_h = 0
    new_img_w = 0
    if (img_h%patch_h)==0:
        new_img_h = img_h
    else:
        new_img_h = (int(int(img_h)/int(patch_h))+1)*patch_h
    if (img_w%patch_w)==0:
        new_img_w = img_w
    else:
        new_img_w = (int(int(img_w)/int(patch_w))+1)*patch_w
    new_data = np.zeros((data.shape[0],data.shape[1],int(new_img_h),int(new_img_w)))
    new_data[:,:,0:img_h,0:img_w] = data[:,:,:,:]
    return new_data

def extract_ordered(full_imgs, patch_h, patch_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    N_patches_h = int(img_h/patch_h) #round to lowest int
    if (img_h%patch_h != 0):
        print("warning: " +str(N_patches_h) +" patches in height, with about " +str(img_h%patch_h) +" pixels left over")
    N_patches_w = int(img_w/patch_w) #round to lowest int
    if (img_h%patch_h != 0):
        print("warning: " +str(N_patches_w) +" patches in width, with about " +str(img_w%patch_w) +" pixels left over")
    print("number of patches per image: " +str(N_patches_h*N_patches_w))
    N_patches_tot = (N_patches_h*N_patches_w)*full_imgs.shape[0]
    patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))

    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                patch = full_imgs[i,:,h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches  #array with all the full_imgs divided in patches

train_set = RetinalDataset('train')
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True, num_workers=0)
val_set = RetinalDataset('val')
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=16, shuffle=False, num_workers=0)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1)

        self.fc = nn.Linear(1, 2)

    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)

        # import pdb; pdb.set_trace()

        return c10

net = Unet(1,1).cuda()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

#Recompone the full images with the patches
def recompone(data, N_h, N_w):
    assert (data.shape[1]==1 or data.shape[1]==3)  #check the channel is 1 or 3
    assert(len(data.shape)==4)
    N_pacth_per_img = N_w*N_h
    assert(data.shape[0]%N_pacth_per_img == 0)
    N_full_imgs = data.shape[0]/N_pacth_per_img
    patch_h = data.shape[2]
    patch_w = data.shape[3]
    N_pacth_per_img = N_w*N_h
    #define and start full recompone
    # import pdb; pdb.set_trace()
    full_recomp = np.empty((int(N_full_imgs),data.shape[1],N_h*patch_h,N_w*patch_w))
    k = 0  #iter full img
    s = 0  #iter single patch
    while (s<data.shape[0]):
        #recompone one:
        single_recon = np.empty((data.shape[1],N_h*patch_h,N_w*patch_w))
        for h in range(N_h):
            for w in range(N_w):
                single_recon[:,h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w]=data[s]
                s+=1
        full_recomp[k]=single_recon
        k+=1
    assert (k==N_full_imgs)
    return full_recomp


predicted_patches = torch.zeros(len(val_set), 1, 48, 48)
for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    net.train()
    for i, data in enumerate(tqdm(train_loader)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda().float()
        labels = labels.cuda().float()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        # import pdb; pdb.set_trace()

        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        optimizer.step()

        outputs = (outputs >= 0.5).cpu()
        acc = float((outputs == labels.cpu()).sum()) / (outputs.shape[0] * outputs.shape[2] * outputs.shape[3])
        # import pdb; pdb.set_trace()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 0:  # print every 2000 mini-batches
            print('[%d, %5d] acc: %.3f, loss: %.3f' %
                  (epoch + 1, i + 1, acc, running_loss / (i + 1)))
            running_loss = 0.0

    acc = 0.
    net.eval()
    for i, data in enumerate(tqdm(val_loader)):
        inputs, labels = data
        inputs = inputs.cuda().float()
        labels = labels.cuda().float()

        with torch.no_grad():
            outputs = net(inputs)
            outputs = (outputs >= 0.5).cpu()
            predicted_patches[i * 16:(i + 1) * 16, :, :, :] = outputs

        if i == 9:
            predicted_patches[i * 16:, :, :, :] = outputs

            predicted_patches = predicted_patches.numpy()
            pred_imgs = recompone(predicted_patches, 13, 12)
            pred_imgs = pred_imgs[:, :, 0:584, 0:565].squeeze(0)
            pred_imgs = pred_imgs * 255
            img = Image.fromarray(np.uint8(pred_imgs).squeeze(0), 'L')
            img.save('pred.png')

            gt_img = np.array(Image.open('data/gt/24.png').convert('1')) * 255
            acc = (pred_imgs == gt_img).sum() / (gt_img.shape[0] * gt_img.shape[1])

            print('[%d] acc: %.3f' %
                  (epoch + 1, acc))
            # import pdb; pdb.set_trace()

import matplotlib.pyplot as plt

img=Image.open('data/train/24.png')
plt.imshow(img)
plt.show()
print("raw image")

img=Image.open('pred.png').convert('1')
plt.imshow(img)
plt.show()
print("predicted vessel image")

img=Image.open('data/gt/24.png').convert('1')
plt.imshow(img)
plt.show()
print("target image")

torch.save(net.state_dict(), 'models/model.ckpt')
torch.save(net, 'models/model.pth')
net.load_state_dict(torch.load('models/model.ckpt'))