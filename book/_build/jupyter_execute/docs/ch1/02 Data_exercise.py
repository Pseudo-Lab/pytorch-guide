#!/usr/bin/env python
# coding: utf-8

# # 02 Data_exercise
# 02 Data 에서 다룬 내용을 코드로 살펴봅니다.

# In[2]:


# !pip install torchvision


# In[4]:


from torchvision import datasets

# CIFAR 
CIFAR_train = datasets.CIFAR10( 
    root="CIFAR",	# (str), 데이터셋이 저장될 경로, True 인 경우 루트 디렉토리에 생성
    train=True,		# (bool), True 인 경우, train set 에서 데이터 생성
    download=True	# (bool), True 인 경우, root 디렉토리에 데이터 다운로드
)
CIFAR_test = datasets.CIFAR10(
    root="CIFAR",
    train=False,	# (bool), Flase 인 경우, test set 에서 데이터 생성
    download=True
)


# In[6]:


from torchvision import datasets
from torchvision.transforms import ToTensor, RandomCrop, RandomVerticalFlip

# ToTensor, 데이터 타입을 tensor 로 변경
cifar_test_ToTensor = datasets.CIFAR10(
    root="CIFAR_ToTensor",
    train=False,
    download=True,
    transform=ToTensor(),	# data 타입을 Tesor 로 변경
)

# RandomCrop, 데이터를 랜덤으로 crop 하여 size 크기로 출력
cifar_test_RandomCrop = datasets.CIFAR10(
    root="CIFAR_RandomCrop",
    train=False,
    download=True,
    transform=RandomCrop(25),	# data 를 랜덤으로 crop 하여 25*25 사이즈로 출력
)

# RandomVerticalFlip, 이미지 뒤집기
cifar_test_RandomVerticalFlip = datasets.CIFAR10(
    root="CIFAR_RandomVerticalFlip",
    train=False,
    download=True,
    transform=RandomVerticalFlip(p=0.5),	# data 를 수직으로 뒤집기
)


# In[27]:


import torch
import matplotlib.pyplot as plt


# 노란 배경의 흰 말 사진(idx :808) 데이터로 확인
crop_image, l1 = cifar_test_RandomCrop[808] #transform 거친 데이터
image, l2 = test[808] # 원본 데이터

fig = plt.figure(figsize=(7, 7)) # 출력할 이미지 사이즈 조정

ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(crop_image)
ax1.set_xlabel("crop, label:"+ str(l1))
ax1.set_xticks([]), ax1.set_yticks([])
 
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(image)
ax2.set_xlabel("origin, label:"+ str(l2))
ax2.set_xticks([]), ax2.set_yticks([])
 
plt.show()


# In[67]:


from torch.utils.data import DataLoader

loader = DataLoader(cifar_test_ToTensor, batch_size=64)


# In[68]:


feature, label = next(iter(loader))
print("feature shape : ",feature.shape)
print("label shape : ",label.shape)


# In[69]:


from torch.utils.data import DataLoader

torch.manual_seed(42)
loader_with_shuffle = DataLoader(cifar_test_ToTensor, batch_size=64, shuffle=True)

feature_with_shuffle, label = next(iter(loader_with_shuffle))


# In[75]:


fig = plt.figure(figsize=(7, 7))

ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(feature_with_shuffle[8].permute(1,2,0))
ax1.set_xlabel("with_shuffle")
ax1.set_xticks([]), ax1.set_yticks([])
 
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(feature[8].permute(1,2,0))
ax2.set_xlabel("origin")
ax2.set_xticks([]), ax2.set_yticks([])
 
plt.show()


# *추가 예정*
