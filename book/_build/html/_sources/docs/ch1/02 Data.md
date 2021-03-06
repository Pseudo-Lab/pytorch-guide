# 02 Data

PyTorch 에서 데이터를 조작하는 방법을 안내합니다.



## 1. Data set

PyTorch 의 TorchVision, TorchText, TorchAudio 등의 라이브러리에서는 다양한 데이터셋 샘플을 제공합니다.

각 라이브러리에 대해 학습 데이터와 테스트 데이터를 사용하는 방법과 더불어 dataset을 커스텀하여 사용하는 방법에 대해 소개하겠습니다.



### 1.1 TorchVision

`TorchVision` 는 컴퓨터비전 라이브러리로 datasets, transforms(argument) 등의 다양한 기능을 제공하고 있습니다. 이번 단계에서는 `TorchVision` 에서 제공하는 Datasets 을 활용하는 방법에 대해서 알아보겠습니다. `TorchVision` 에 대한 자세한 설명은 관련 파트(링크) 에서 자세히 다루도록 하겠습니다.



우선 torchvision 을 설치합니다. 환경에 따라 아래 명령어로 설치할 수 있습니다.

1.  Anaconda 

   ```python
   conda install torchvision -c pytorch
   ```

2. pip

   ```python
   !pip install torchvision
   ```



이후 torchvision 에서 여러 vision dataset 을 다운받을 수 있습니다. 대표적인 vision dataset 인 CIFAR 를 예시로 살펴보겠습니다.

```python
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
```

`datasets` 에서 사용하는 매개변수는 아래와 같습니다.

1. root
2. train
3. download
4. transform
5. target_transform



이때, `torchvision.transforms` 클래스를 사용하면 데이터 타입을 tesor 로 변환하거나 이미지에 변형을 줄 수 있습니다. 관련하여 자세한 내용은 Data Argument 파트(링크)에서 자세히 다루며, 여기에서는 가볍게 살펴보도록 하겠습니다.

```python
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
```



transforms 중 RandomCrop 의 결과를 살펴보겠습니다. 원래 Cifar 데이터의 크기는 32 * 32 이지만, RandomCrop(25) 으로 25 * 25 크기로 잘려 변형되었습니다.

matplotlib 라이브러리로 시각화하여 결과를 살펴보겠습니다.

```python
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
```

![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAdAAAADnCAYAAACuVQehAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAAhWElEQVR4nO3dWYxk133f8d+prfdleno4w+Es5FAmKVIbJYWUFMkRHNkBnMhOFMGKYyhOkOUhL0lsBXlIoiAGAiQwYgRB4BiBYcjSg5AosGAlSBDYjhbLJsNoJSmNKIqTGQ5n7e7p7uru2qtOHqZva8z5/8+dOhyLw+nvByA4OKfuubeq69ape+v/P/8QYxQAABhP5fU+AAAA3oiYQAEAyMAECgBABiZQAAAyMIECAJCBCRQAgAy1cR68vNiIJ49O+g+IZfNxScpMSPeHktFvCVk7+BH4+ve2VmOMh17v4/AsLzbi/ca5nM5q8zrtM7M/qrojVStDs73ineTJzxbnuJzPk6zPET439q3UuTzWBHry6KSe/swTbn8cJCZXSQqDdHdtlOyvjG7lXVwyCad3AdwW1Sf/4NzrfQwp9x+d1Nc+856b2rtD/xyNzoRUifZEebE17461NL1ttk9V7RM0+dnifK54nyfpzxFn0uVzY99KncvcwgUAIAMTKAAAGZhAAQDIwAQKAECGsYKIANxFjNifQd8P1lnbXjLbl+6ZMNt//4t+FO4jpx4w248dsfc/GNn7kKRKWDPb75t+0WyvV3vuWO41RUmGAPYnrkABAMgw1hVofzSjC60n3f7Ut0TJ/6ZY8L4xFtLfHPf2ku7mmyQA4DbgChQAgAxMoAAAZGACBQAgAxMoAAAZSGMB9qt48wKvw6G/6OuV/lvM9sbUnzHbt+M33bGev3TAbD+zXjfbQ2IJ+PnpU/Y299kL1p9ceMEdS87TJ/YQFq5AAQDIwAQKAECGsW7hbrUn9ZXTD7v9qdsskn+rZW9755ZLIXnrpVBSdohbMQCA24ErUAAAMjCBAgCQgShcYN+6+SeXCTsIVpJ07NSb7VGmHzXbl4+uuGMN+g2zfX1102wfDXfcsaZPHDXbV3rvMNsP91/xx6o6+0n/OoV9iitQAAAyMIECAJCBCRQAgAxMoAAAZBgriGg4jFpf93M1Uz/0S/6P/QXvR/9C6sf/vX14QQAFggEAALcBUbjAfmUsKhIqdnSsJM0fsNev3ezbq5c0t/wv2ztt+4vuMNrfcLvdqjvW2qY91uSUfbwrk/e7Y52cfc7uiN7+WZllP+MWLgAAGZhAAQDIwAQKAEAGJlAAADIwgQIAkIEJFACADGOlsYwU1HbCzKV0qLnkh5sXvLDzQir8vOCGoRfccPS9B5TuA7grxJvTT0aNZffh1ZnjZvvOpb7Zvrbedcfq9OzPgumpKbO9H/2PqqvrLbN9OBqY7XP1H3PHOjr7ktleV9vZgsTy/YwrUAAAMjCBAgCQgQkUAIAMTKAAAGRgAgUAIAOLyQP7UpR082Lvo/qSu8UoLpjt//v3P2+2X7h4yR1raWnGbO91Nsz2et2P0O8O7cXsV9bs6NzvT0y6Y71p+aTZfrTxPbM9Jq5Bguzj8iTj/51g35DIilCwjy3Kjk5OBRRH5+i8/Yfovy6jYL8uo4yA5qq3f/fJJF7ljAQMrkABAMjABAoAQIaxbuFGRQ2jc/mvdLKz5Cc8F7zE50IqAbrgJUIX/IToAonRAIByXIECAJCBCRQAgAxE4QLYUxl13L5Xzl002//oq98022uTfuRsrTZvtg/622b7/Iy/Rm+nb18H9Ht2tOfGtZ471rnVt5jty/edM9vr8l+v6EaI2uGeMaTCQO2xkoGjFfvY4sheD9wJji0Gs8cK9nENw80R3qWiE+mbel3cSGPvpzj/J7rky+/gChQAgAxMoAAAZGACBQAgAxMoAAAZxssDHQ3dpbak9HJbkr/kVsFbequQWoKr4C3FVfCW5CqkluaSxl+e6+bxS5SkoSaX7tp7UPo5uEt53eIxeMt67W1ecoypZb4kf6mvH/Ynu2+JtwRYwV8KrEDhdWC/4woUAIAMpLEA+5WRglCJdhqJJJ0587LZXp+0U0zm52bdsUYDe0Ww4dC+OxKjn3pSr0+b7ZVgf7w1qv6drks79l20a8MjZvvh2hl3rLI7LcYWiT7njkf079aMKnN2++TDZnullvh7Ve3UF3Wv2GO17PfK9eOyCwkM6/eZ7aG/4o5VG9qpVdVKxmLyGbgCBQAgAxMoAAAZmEABAMjABAoAQAYmUAAAMoyXBxpHGjjRc1J6wWfJX/S54C3+XEgtAl3wFoMueItCF1KLQ0v+AtEFb6Hove1LVyxOj39LMWTOItJ7YziLSe8dQWmqa0meqbPAdCFroek/sYOSPNRbWRW6NFe2LA/0bqgbe/PrNBz5udbbbTtCdmbejoKNse+O1WnbkbAjp95wr+9/7tQn7OjRiYmGvUHD/9jb6tvnxnrnAbP98Gzq88R7j3kRtYnzwnlPD2bs6GBJqiz+dbt97l327qv1xP6d93t30z6ubT/ffrtj/43Prxyydz267I71YzP/xWyfdrdJTXnjR+hyBQoAQAYmUAAAMjCBAgCQgQkUAIAMTKAAAGRgLVxgvzICK4eVeffhOx07ErO5aa+fOzPZdMfqOFG1jSl7/6PoR45XqvZx1Zz1WxsNO2pYkob9CbN9u21HiA7n/ZD1qhNVG52qVLFur10rSaMDH7I7lv6Cv82kvbasKvbH/nDoRwF7AdX9kZ150a4+4Y713Hk7cvnF08+b7TOzC+5YUyefNNsfnP1de4NEpHNN6ewEC1egAABkGPMKNLrVEqR0xQTJr5pQ8KonFFJVFApeNYWCV1WhkKquIOVUWLhphJL+klykRPWFgleFYa/fqcZQSFVlkBKVGQpOhYa98ROVGiS/WkPBq9pQSFVvKHhVHAp+NYcC9UCB/Y4rUAAAMjCBAgCQgQkUAIAMTKAAAGQgjQXYl4Ks78/d6KcMvHTOTkuZq26Z7T/5zjV3rC9+2w6G60c7jSWRYeHGc1WdYLda1f/Y83ZzoWmnaxxdetQda2nKfr0q9UWzvZpISYkL7zXbB9EPzKyMnNyTkf0s2y0/SLO9YweIdtv2PrZ2/IIWZ1+2F3q/eMFZAD76i8l3m/b7demdh832Qw1/rJhxPckVKAAAGZhAAQDIMNYt3NEoqtfzcz1TNfskv25fwa3fV0jU8St49fwKXl2/Qrq+n1T+naMsj7OkFmZJLctU/b+CVwdwr9+pB1hI1gWU/NqABadGYCFVK1Dy6wUWvLqBhVT9wIJXR7Dg1xMs8OsHsN9xBQoAQAYmUAAAMnAfCti3bv7+HHXQffTKFTuq9pHDL5jtf/7t/sfL9y8eN9tfXrMXc4+xbGnFm1Wq9vVBz4lClaQo++eDjc6i2f7i+kfcsU5O2xGiB5buMdsnZv3X3vtlKA4SP6kN7I2Cc9nU2vFfl9VVO0K32+2a7evrO/5Ya/Y2jWl7Cc+tjbPuWM+/bD//x47ZP3UdOn7VHStndU6uQAEAyMAECgBABiZQAAAyMIECAJBh7Hqgg0Q90FTVeMmvHL93MCV1JlOV5AteRfmCV1l+b/tEhXnJrzJf8KrN7/Unqs5LicrzhcRyX3tjeJXoC05F+kKqMr3kV6cveFXqC6lq9ZJfsb7gVa4vpCrYF7xK9gW3on2hLJ8XwF2PKFxgvzLCMc+d89evXbtqFyF/8+P2uqczE/6X2XsPTprtlzbsRTxiVv1y+wv7IPUF0dlPzblXtz1Ycoe6sj1ltjcOOAvGJBap8davVeKipd2xo129L8hra/bavZK0smIvjjIY2RdUm01/Ldz1Tbvv4IJ9gTQREuszv7Jitr9wYdFsf/yY/56cVuL1d3ALFwCADEygAABkYAIFACADEygAABmYQAEAyMAECgBAhrHSWKKkUSKcvCR9sHSx3mpJHmitWn64ZYdwoZnOUTy69Giyf2nKD/WWpEp9MdlfLcnjjAvvTfYPYnkubGVUkqiZWExbktote+Hovf6ddLh3t53e/9aOH+IuSWdfTtfivHihpFZnLK8H2m2mc0WX3nk42X+oUb6PO1oIkpGXffH8t9xNLp47b7YfXJh39uHnRC8v2ikmdSdfpFr3z/2Kl9fsLEAfR/5xeZ9B3ja1qv9er1eclAkn9aPTTn1AOs9l6O9/fWPbbL+2brevX/Pr+Lba9jGHin1cK2sb7lg7LXv/hw/Yjz9yr5+7f8UpcHDuqv0+urS16I714Pz45zRXoAAAZGACBQAgAxMoAAAZmEABAMjABAoAQAYWkwf2pSjFmyNLH7t/3d3iox+0I1QPLXgLnfvR3kvzdiT2xOQ9dnvDXphdkqo1+2OsUrGvD0ZDP9rVeEkkSf2+3dFu2Qu2S1J32l4wv9u1xwp9v9KVVwSrtePvf6NpR7t+/6VLZnsqCndy0o7+D0ZBAklaWfXH6vV3zPa5mv3eO7J00B1r6eCM2b7W3DLbX7jsZ2GcmLUXpk/hChQAgAxjXYEGBVW83CZJ0cm7ulWVano+75XkL0pSlP8tTpI2OovJ/hfXP5LsPzmdzh88sGR/gy5MzPrfpiSV5srGQXnJnd4gPYjzpXFPayf9Oq+upvNEu13/W7Ekra/b30D3xl9Lb9+Ytr91FrY2zib7Jen5l9Ov42PHjiT7Dx2/WroPAHc3rkABAMjABAoAQAYmUAAAMhCFC+xHUdLo5piFY4f8OIaP/4wdVTrbcDZIhCMcnr5mty+9yWzvRj/2otFwDsCNyfDX3I5O+MDQiStoNv3f82cm6/Y2m/bHbq3uv/Y72/aLubZmR9pKUtdZJ/f8Bfu17/X9dXVnR/a1VnQimncS62XHkR3jcHDSjg6eb/gxGQsLdkzKTseO8n5lzY9BaR6fc/s8XIECAJCBCRQAgAxMoAAAZGACBQAgw3gLKYSgetX+YVzyf4AfYw/J3kFpxW6VLkTg1Ovdsz1YSvZf2faXFJOkxgEvomJXr2QhhLLFImK66LgktTvphQiGJa/j2lq6aPjKir9MlyQNnILBhc1muqD2+ma6/+BCuqj4REgvdiFJL72SXrbrhQuLyf7Hj/lBLQD2B65AAQDIQBoLsF8Zd2tS9zcW7SwWBe97eGJpzwOT9nKQD55YNNvPrvp3HWKwP8YGzgLwqRtdAycto9myj3cusbRpc8tOcak7m0zPOi+wpJUVO11lNXG3qOvczVrbsO8gTTmL319nvy69oX1HrdtvJ8ay7zAtz9mv1/zEmjvSzPR9ZvuBGft1abb9VJXL24tun4crUAAAMjCBAgCQgQkUAIAMTKAAAGRgAgUAIMPYBbXriUTKaj09XKVSsruSgtxx5ETV3XgM1XSeZNkYtaq/oLIk1RNRd5KkkhzITrssWbbkNXAWiL7R+oa/wLQkXVtP969fS+d5ttrp5xgq6eewsraR7N9ppY/v8IFkt47ceyj9AElXrviRfZJ07mr6u+WlrcXSfdzxjLdiJZlIXZ6D/CfH98+1xqQd8XnixAmzfaPnv6euNe285xjt92lIvD3bfXusgRNVWm/4g21t2++hqRn7c7A79M+ryyurZntzy8/53u7YnxWdnh1RPDmZ+PuO7PdFt2vvv9X3890bDfv8Xl60j2thwh+rWtsw22fmZ832Ztd/jpuddI6/hStQAAAyMIECAJCBCRQAgAxMoAAAZGACBQAgAxMoAAAZWEwewA8lslhCsDujkxeSKm9YqdlpLHNTdjnBxQU/fWvVSdsaOSllw0R6TW+wZbZ323aKRSpzr161X5fVa/bjt1v+AuxbTbuv0/FLEzY79ja9gb1oeydRRXA0Yy/CPhjYr3F/4A/24GE79eXYov03nqn5aW2n5q6Y7ac7bzLbQ98vNzlQSSlKw3h5oJWgiUk/D3Kikc6jqdbK8kTTF8Qjp1LCjRLnhiSp71Vo2NVupWtpdpMVC6RuNz1+6KdzKBOpYJKk1k76+CRpo5nOo/z+S5eS/WV5oJOT6XqcIaT/jiur6fF7ffsEL8zV1pP9R5YOJvslaengTLJ/rWl/kBZeuLxcug8Adzdu4QIAkIEJFACADEygAABkYAIFACADUbgAfihdB+C2jeUVRYjRbp+a9wMUBwN7sfG2E4U6in7k6k7LXrR9e2vDbJ+d84Mqe0P7mDcu20F+vX55kOSrtZ1F7q/32cF6262rZnuj4gff9QeL9j469j4GAztqWZJqTkGSc9fmzfaZSf85Pnb8B2b7yvZhs/1a56g71vp2SZUKA1egAABkYAIFACDDWLdw67Wgw0t+smk3pmtlNholiaol9UBvpR5hKnlbkoaD9AOazXQO4sxkPb39ZvolrdXTz3FnO50IuraWzvGUpG5JzdDzF5xM7l29fnr72VH6e1csydfd2fHr+0lSHKVzXQ9OpvNY5xv+LbrCwsJCsn8nlVUu6ZW18lxTAHc3rkABAMjABAoAQAaicAG8Nt4de2eNXEkaDuzIyn7X/okiDv2fbzrb9tKOrY79c8yw4n/stVr2Nl7k6ije547VdtbPbTntCv7PQ8OR/SK3EmvObm7ZP9Wc+cF3zfbOoVPuWNNT9tKX7U7TbB9G/2ea0+ftpUCf++4hs31U9Zf+/OQvbJjtbz3+ktl+pW2v6StJq63jbp+HK1AAADIwgQIAkIEJFACADEygAABkGCuIaKJR04MnFt3+s6vpOpExpHc3KKnVeSvLjA1KchCbLX+JKUmaq6RzWZtb6TzRenpzTc+m64murKTzPFfX7B/tb9QdpfMg1zbS9TinSmqeJqsuS+oN03me3cTyY9elczCX59J/g/mJtZLxpZlpP/hDkg7MpF/nZiIYAcD+wBUoAAAZSGMB8BrZdyRiYmWxWtiy23XebL980X68JF25ai8oXqnaaSG1up2SIUmjrnOHKtrXGu2Wv2pWbNj7rziXLaPEnZ2Bk66y01xxt+n0Lpvty7P2HRonI0WStLphpxF1nDt+tcTtwp2OPe30q0tm+5T8O071ob1y24P3nDXbwyn/mvGrL7/b7fNwBQoAQAYmUAAAMjCBAgCQgQkUAIAMTKAAAGQYKwq30WjoxIkTbv9GL52oea2ZrvMYY7oWZmJt6j3tfnofg5IcxHojvZOt7fR3jqmZ9EvadaLGCpdXVpP9za3085Ok7U66nmenl86FnZwsqbvqLGxd6HbTx9jqp/NEG410LuzyYvr4FybS40tStbaR7J+Zn032N7vltWnhn0vViv0e7bavmO1rq/57quVsMz1tR3Uq+nnOg4H93ur37P33+v57tV6zn//QydPu9Pzz9tr6htm+2fJr4/7sk/YxP/nR+832M2f9MNwXrl0027/wjP15GKr+azw/bT/PXrDPqR9/2M8LP7ls/73i0H7tHzj0gjvW2qAs//1mXIECAJCBCRQAgAxMoAAAZGACBQAgAxMoAAAZWAsXwGsSne/hUX5VoIEWzPatll3lptn018Jtd+wozek5Zy3ahl81aii7IlSns2G2b276UbD9rr1NcLINen0/ujwM7QpK73/Ij1D98Nvs1/9Aw47OfeQePwr14VV7P1/6hr1O7XMX7ddekt71kJ3JcWDOPq43H1t3x6rV7MjdwcD+O1ajP+U9uvyS2+fhChQAgAxjXYFWKlXNzTm5VZIWF9L5h6sb6fy+0SidIzmMJfVCJfUG/jdVSeq20zmE9ZJXpF5N54muXktvv91K56FuNdP9nU661qckNTvpMXqDdD1N5wv9ntFMuhbmYJD+O/adyhKFBw+n80iPLabfZzO19PtMkk7N2bmDhdOdNyX7Q79Rug8AdzeuQAEAyMAECgBABiZQAAAyMIECAJCBNBYAfypG9UW378zqm832Z77jBPkN/aCvBw7ZQWlTTlDjRtcPRlyes68pHjniFCgY2YusS9LctJ3KcWDWDkScqPvBd0uz9v4fPeY/l6VJ+/kPR/b+KyN/rAcO2dv8zPumzPaX/6cfbHnxir2fUxN2YN7RZXcoyTnmanSuDaN/XNOV8kIdr8YVKAAAGZhAAQDIMOYt3KgY/Ry8qXn7cr4wGKTrNLZL8hdHsTwHcqeVrqe5vbWR7J+dm0j294bp57hxOZ2D2Ouna2mWaZfUM73+GHvVksJ262qyv1E5mOzvDxbT+++k9+/VXCzUaunvdeeuzSf7ZybLX6PHjv8g2b+yfTjZf61ztHQfAO5uXIECAJCBCRQAgAxE4QJ4TUKwo0c39B53m6+fsRd0rw1Om+0fePiyO9axRTt68mrL/inh81+zF7KXpHsX7J+JPv5BO9pzlFgAvtGwr0+mnJ8oKhX/57GK7J9+KsGPKvV+LKp4m0T/56WKE9X60++2t3n25Rl3rKdOz5rtI+fnrdmJ1DTlLAvqHG8I/nMMSi/TauEKFACADEygAABkYAIFACADEygAABmYQAEAyDBWFG6MI/W7/kIBcVhNbt/ZXk/2tzrpQs/DSvnhtlrpMcoWERjF+5L97ZKC3K2SfgV7fczCcJReaKFVUoxakja30lW9z/zgu8n+zqFTyf7pKT/CTpLanWayfxjTC2qcPm9HaBae++6hZP+oml7IQZI++Qsbyf63Hn8p2X+lnS4qvp8EJ6zz/Jr/PomDs2b7T73dbj8y779n6lU7enV62j4PZur+QiEj2WMtTtjRubXJ1PlqR+5Gdz3WVEStv06tzxnPOeREQK+GziI2987Zx/Wht/mD/eCC/Tm+MGMvYjNTT7zG3pq34wfUEoULAMCPChMoAAAZmEABAMjABAoAQAYmUAAAMjCBAgCQgcXkgX1rvNq03qO99Ide81vuWN0tO+VtbsJemD51pDt9OzVsqmYvMj/T8Bdt747sJ5NVxdfdaPzRKjnXOl66yvgjKTj7957JW475tZsfPGr/vaan7PS1iXriiJ2UoOClCiUWk88x1gQa1FNN593+yxe3kttfuZouYlyppnMka/V0/qEkjboleZhe3tCudss+6fY2b6SPsVLyPh+VnDyDkjzPneZKegeSOj2/coUkLc+upbdPp3FqdaMk33eYfo61klN4p5N+W/arS8n+KaVzgSWpPrQ/qAsP3nM22R9OcfMG2O/4FAAAIAMTKAAAGZhAAQDIwAQKAEAGonCBfcsI9krEf4XgfN8e2QuKnzhw0R3rqda9ZvuzF46a7ZMVPzDs2dN2YN1f+nMNs31h2g9iu+jUIfCCA91oz+u9Tqv3IqeC78ZbGP52c5+ls/+lOT8K97577CDE9Z222T5UIuDPeU+6Qbj+SIm/i48rUAAAMjCBAgCQYbx6oKO+uu0rbv/aajqHspXYVpKmp9P5fYqT6X5Jg0E6D7TfSx9jr+/XO5Wkei2dwzgc+bcuJKnT85O4Jena+kayf7N1KdkvST/7ZPo5PvnR+5P9Z86mE0FfuObfmpOkLzyT/l4Wqum/4/x0+jXqhXQe6o8/XF4z9eRy+n0Sh+m/8wOHXijdB4C7G1egAABkYAIFACADUbjAfmWsC+qteXqdHaUYnfVjj8z7P4e8/SE74vLb5x4y27c3nfBYSf/rK6fN9pP3zpntJ47Y7ZK0sm0fl6K7ErA71vghsjmr1N6ZKomnMuxfM9svXrZfrxfP29HUkrT8iP33is5Paen39/i4AgUAIAMTKAAAGZhAAQDIwAQKAECGsYKIhqOqtlr+D/DNZroeaLuTzs+bniuptdmwC67eaCh7WbFCp7OR7N/cTOdZ9rvp7UNM15ns9dP5h2HoB0tI0vsfKs9x/PDb0rmoBxrpPNFH7knnaT68mj6GL30jXY/zuYvpv/O7HjqR7D8wlz7+Nx9bT/ZLUq2WziUdDNLvo2ok/g7Y77gCBQAgA1+jAbw20f4eXhn5aRyPP7Bmtq+37FWwXtzw7wicOrRgdwzt9Id77nEeL+neNfsOUTBSfiS5z323M9F3d/CyVUaJRfa3W/Ydsqsr9p2lp7/j3/F67yPegv0/mpQgrkABAMjABAoAQAYmUAAAMjCBAgCQgQkUAIAMIbqLJBsPDmFF0rk/vcMB7honY4yHXu+D8HAuA7fMPZfHmkABAMB13MIFACADEygAABmYQN+AQgj3hxCeL3nMB0MI/33Mcb8UQni30f6fQwjf2v3vbAjhW2MeMnBXCCH8jxDCYsljfiWE8KHM8Tm330BYyu9HJIRQi7Fkpfk7VIzxY8W/Qwj/VlJ6xXvgLhNCCLoeM/LTZY+NMX7yR3BItwXn9mvDFehtEkL4GyGEZ0MI3w4hfGa37VMhhF8LIXxR0r8JIbwjhPD07uM+H0I4sPu4L4UQ/l0I4Y9DCM+HEJ4YY7/3hxD+MITwjd3/3ndD9/zufr4bQviNEEJld5ufCiE8tfv4z4UQZm9xX0HSz0n67K0eH/BGEEL4pd1z7/kQwj/cbbs/hHA6hPDrkr4h6fjuVdrybv8/DyF8L4TweyGEz4YQPrHb/qkQwkd3/302hPAvd8+150IIj4xxTJzbdzgm0NsghPCYpH8q6SdijG+X9A9u6H5I0odijL8s6dOS/kmM8W2SnpP0L2543EyM8X2S/r6k3xpj91cl/WSM8Z2SPibp39/Q94SkX5b0VkkPSvrI7sn/z3aP6Z2Svibpl4zn9JvGLZ8PSLoSY3xxjOMD7mghhHdJ+luSnpT0Hkl/N4Tw+G73w5I+HWN8PMZ47oZt3i3pr0p6XNJHJN10e/QGq7vn2n+U9IkxDo1z+w7HLdzb4yck/dcY46okxRiv3dD3uRjjMISwIGkxxvjl3fbflvS5Gx732d1tvxJCmA8hLMYYN25h33VJ/yGE8A5JQ12fsAvPxBjPSFII4bOS3i+pI+lRSX90/UunGpKeevWgMca/Y+zr58U3VNx93i/p8zHGHUkKIfyOrk8oX5B0Lsb4tLPN78YY27vb/LfE+L+z+/+v6/pke6s4t+9wTKC3R5BfuyhdXfqHXr39rSbo/iNJVyS9XdfvKNxY+8caM0j6vRjjz9/i+JKu/4ar6yf/u8bZDngDSNW+8s7fceplFXW6hhrvM5dz+w7HLdzb4w8k/VwI4aAkhRCWXv2AGOOmpPUQwgd2mz4u6cs3PORju9u+X9JmjHEzhPBECOHTJftekHQpxjjaHbN6Q98TIYQHdn8f+Zikr0p6WtKfDSG8aXd/0yGEh149qOFDkr4XY3zlFh4LvJF8RdJf3j0XZiT9FUl/WLLNVyV9OIQwufs7418cZ4ec23cHJtDbIMb4HUn/StKXQwjflvRrzkN/UdKvhhCelfQOSb9yQ996COGPJf2GpL+923ZCUrtk978u6RdDCE/r+i2eG78xPyXpX0t6XtL/0/XbVCuS/qakz+4ex9OSbgpsMH4n+WviFg/uQjHGb0j6lKRnJP0fSb8ZY/xmyTb/V9dv8X5b12/Rfk3jRbBybt8FWMrvDhBC+JKkT8QYv/aq9l+V9JkY47Ovy4EBcIUQZmOM2yGEaV2/iv17u5PxrWzLuX0X4DfQO1iM8R+/3scAwPWfQgiPSpqU9Nu3OnlKnNt3C65AAQDIwG+gAABkYAIFACADEygAABmYQAEAyMAECgBABiZQAAAy/H81CerYhVyaKgAAAABJRU5ErkJggg==)

위 출력 결과에서 crop 된 이미지가 사이즈에 맞춰 변형된 것을 확인할 수 있습니다. 참고로 cifar 10 데이터 셋에서 말 사진은 label 7로 분류됩니다. 또한, dataset으로 불러온 데이터들이 feature(데이터셋의 특징, cifar 데이터에서는 이미지 픽셀값)과 label 로 구성되어 있다는 것을 알 수 있습니다.



불러온 데이터 셋을 `DataLoader` 클래스로 묶어서 객체로 활용할 수 있습니다. 관련 내용은 이하 `DataLoader` 클래스 파트(링크)에서 자세히 다루도록 하겠습니다.

CIFAR 와 MNIST 외의 다양한 vision 데이터가 있습니다. 관련하여 공식문서 ([링크](https://pytorch.org/vision/stable/datasets.html)) 를 참조바랍니다.

---



### 1.2 TorchText

`TorchText` 는 자연어처리 라이브러리로 Tokenization 을 비롯한 다양한 기능을 제공하고 있습니다. 이번 단계에서는 `TorchText` 에서 제공하는 Datasets 을 활용하는 방법에 대해서 알아보겠습니다. `TorchText` 에 대한 자세한 설명은 관련 파트(링크) 에서 자세히 다루도록 하겠습니다.

*(추가예정)*



---



### 1.3 TorchAudio

`TorchAudio` 는 오디오 라이브러리로 ??? 을 비롯한 다양한 기능을 제공하고 있습니다. 이번 단계에서는 `TorchAudio` 에서 제공하는 Datasets 을 활용하는 방법에 대해서 알아보겠습니다. `TorchAudio` 에 대한 자세한 설명은 관련 파트(링크) 에서 자세히 다루도록 하겠습니다.

*(추가예정)*



---



### 1.4 Custom Dataset

사용자가 직접 Dataset을 정의하여 사용할 수도 있습니다.

*(추가예정)*

- https://tutorials.pytorch.kr/beginner/basics/tensorqs_tutorial.html

- https://sanghyu.tistory.com/90
- https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
- https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html#id9



---



## 2. Data load

모델 학습 시 작은 양의 데이터를 사용할 때에는 모든 데이터를 한번에 처리할 수 있지만, 데이터의 양이 많아지면 처리에 어려움이 발생합니다. 

PyTorch  에서는 `torch.utils.data` 의 `DataLoader` 클래스로 데이터 셋을 batch 단위로 끊어 모델 학습에 전달하는 등 데이터를 나누어 관리할 수 있습니다.

이번 파트에서는 PyTorch 에서 모델 학습 시 데이터를 불러오는 방법에 대해 DataLoader 클래스와 함께 살펴보겠습니다.



### 2.1 DataLoader

우선, DataLoader 클래스의 파라미터를 안내하겠습니다.

#### 2.1.1 Batch

Batch 는 한번에 처리할 데이터 양을 의미합니다. 모델 학습 시 전체 데이터를 Batch 단위로 묶어 처리할 수 있습니다.

DataLoader 클래스에서는 `batch_size`  파라미터를 사용하여 Batch 의 크기를 조정합니다. 이때, 데이터 셋의 총 크기가 100일 때, batch_size 가 10 인 경우, 10번의 iteration 을 통해 모든 데이터를 거칠 수 있습니다.

`batch_size`  파라미터 값을 64로 설정하여 dataloader 객체를 정의하겠습니다. 주의해야 할 점은 각 데이터가 tensor 형태여야 합니다. transform 으로 tensor 로 변환하여 데이터를 가져왔던  `cifar_test_ToTensor` 데이터 셋을 DataLoader 로 가져오겠습니다.

```python
from torch.utils.data import DataLoader

loader = DataLoader(cifar_test_ToTensor, batch_size=64)
```



loader 은 train 데이터 셋의 feature, label 과, test 데이터셋의 feature, label 의 batch 단위의 묶음을 가져옵니다. 따라서 loader  의 shape 는 다음과 같습니다.

```python
feature, label = next(iter(loader))
print("feature shape : ",feature.shape)
print("label shape : ",label.shape)
    
>> feature shape :  torch.Size([64, 3, 32, 32])
>> label shape :  torch.Size([64])
```

이때, feature shape 는 `torch.Size([64, 3, 32, 32])` 로 64 는 batch_size, 3은 channel 수로 cifar 이미지가 3개의 채널(RGB)로 구성됨을 의미합니다. 또한 32, 32는 데이터의 크기가 32*32 픽셀임을 뜻합니다.

label 의 크기 역시 batch_size 로 64 크기의 int로 구성되어 있음을 확인할 수 있습니다.



#### 2.1.2 Shuffle

shuffle 은 데이터를 DataLoader 에서 섞어서 사용할지를 설정하는 파라미터입니다. 이때, test 데이터 셋에서는 shuffle 값을 false 로 설정하는 것에 유의해야 합니다. 또한 모델 학습 시 seed 를 고정하여 재현성을 유지하기 위해 `torch.manual_seed` 를 설정합니다.

DataLoader 클래스에서 `shuffle `  파라미터는 bool 형태로 True, False 값이 사용됩니다.



 `shuffle `  파라미터를 새로 지정하여 dataloader 객체를 정의하겠습니다. seed  또한 고정해두겠습니다.

```python
from torch.utils.data import DataLoader

torch.manual_seed(42)	# seed 를 42로 고정
loader_with_shuffle = DataLoader(cifar_test_ToTensor, batch_size=64, shuffle=True)

feature_with_shuffle, label = next(iter(loader_with_shuffle))
```



8번째 이미지를 기준으로 shuffle 된 데이터 셋과 원본 데이터 셋을 비교하여 Shuffle 의 효과를 확인해보겠습니다. 

```python
fig = plt.figure(figsize=(7, 7))
ㄴ
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(feature_with_shuffle[8].permute(1,2,0)) # shuffle 된 데이터셋의 8번째 batch 데이터 셋
ax1.set_xlabel("with_shuffle")
ax1.set_xticks([]), ax1.set_yticks([])
 
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(feature[8].permute(1,2,0))	# 원본 데이터셋의 8번째 batch 데이터 셋
ax2.set_xlabel("origin")
ax2.set_xticks([]), ax2.set_yticks([])
 
plt.show()
```

![img](data:image/png;base64,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)

같은 index 의 이미지이지만 서로 다른 이미지가 출력되는 것을 통해 shuffle 을 통해 데이터가 섞였음을 확인할 수 있습니다.



#### 2.1.3 num_workers

- 추가 필요, 참조 자료 : https://jybaek.tistory.com/799
- 멀티 쓰래딩 관련하여 작성할 예정



이 외에도 sampler,  drop_last 등이 사용됩니다. 관련하여 공식문서 ([링크](https://pytorch.org/docs/stable/data.html)) 를 참조바랍니다.



---



### 2.2 custom DataLoader

사용자가 직접 DataLoader 을 정의하여 사용할 수도 있습니다.

DataLoader 클래스는 크게 3가지 파트로 구분됩니다.



#### 2.2.1 init



#### 2.2.2 get item



#### 2.2.3 len



- https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html
- https://wingnim.tistory.com/33



---



## 3. Data Argument

- https://pytorch.org/vision/stable/transforms.html
- https://tutorials.pytorch.kr/beginner/basics/transforms_tutorial.html













