# MPS-FashionMNIST
Binary and Ternary classification of shoes.

## Dataset
Our classification task focuses on three class of the Fashion-MNIST dataset: **5 (Sandal)**, **7 (Sneaker)**, **9 (Ankle boot)**.  
Each class contains 6000 training and 1000 test data.

![Image 1](https://raw.githubusercontent.com/lmcinnes/umap/master/images/umap_example_fashion_mnist1.png)

## Binary (class 5 & 7)
- Data: **dataset/fdata_57.npy**  
shape of fdata_57.npy: (14000, 28, 28)  

- Label: **dataset/flabel_57.npy**  
shape of flabel_57.npy: (14000)  
*notice: the labels are changed to **0 & 1** instead of the original 5 & 7*  

**Code example of loading the data into torch tensor:**
```
import numpy as np
import torch

x, y = np.load('fdata_57.npy'), np.load('flabel_57.npy')

x_train, x_test = torch.tensor(x[:12000]), torch.tensor(x[12000:])
y_train, y_test = torch.tensor(y[:12000]), torch.tensor(y[12000:])
```

**Both data and label are placed in the order of class 0 & 1.**  
```
x_train[:6000]: class 0, x_train[6000:]: class 1
y_train[:6000]: class 0, y_train[6000:]: class 1

x_test[:1000]: class 0, x_test[1000:]: class 1
y_test[:1000]: class 0, y_test[1000:]: class 1
```
   
## Ternary (class 5 & 7 & 9)
- Data: **dataset/fdata_579.npy**  
shape of fdata_579.npy: (21000, 28, 28)  

- Label: **dataset/flabel_579.npy**  
shape of flabel_579.npy: (21000)  
*notice: the labels are changed to **0 & 1 & 2** instead of the original 5 & 7 & 9*  

**Code example of loading the data into torch tensor:**
```
import numpy as np
import torch

x, y = np.load('fdata_579.npy'), np.load('flabel_579.npy')

x_train, x_test = torch.tensor(x[:18000]), torch.tensor(x[18000:])
y_train, y_test = torch.tensor(y[:18000]), torch.tensor(y[18000:])
```

**Both data and label are placed in the order of class 0 & 1 & 2**  
```
x_train[:6000]: class 0, x_train[6000:12000]: class 1, x_train[12000:18000]: class 2
y_train[:6000]: class 0, y_train[6000:12000]: class 1, y_train[12000:18000]: class 2

x_test[:1000]: class 0, x_test[1000:2000]: class 1, x_test[2000:3000]: class 2
y_test[:1000]: class 0, y_test[1000:2000]: class 1, y_test[2000:3000]: class 2
```

- You can customized another binary dataset comprising only class 7 & 9 or class 5 & 9 from this ternary one.  
Judging from the UMAP above, I assume that class 5 & 7 will make the problem complex enough.

## Notice
***Since the data are placed in order, you have to shuffle them first before training***
