# MPS-FashionMNIST
Binary and Ternary classification of shoes.

## Dataset
Our classification task focuses on three class of the Fashion-MNIST dataset: 5 (Sandal), 7 (Sneaker), 9 (Ankle boot).  
Each class contains 6000 training and 1000 test data.

![Image 1](https://raw.githubusercontent.com/lmcinnes/umap/master/images/umap_example_fashion_mnist1.png)

### Binary (class 5 & 7)
Data: dataset/fdata_57.npy  
Shape of fdata_57.npy: (14000, 28, 28)  

Label: dataset/flabel_57.npy  
Shape of flabel_57.npy: (14000)  
*notice: the labels are changed to 0 & 1 instead of the original 5 & 7

Code example of loading the data into torch tensor:
```
import numpy as np
import torch

x, y = np.load('fdata_57.npy'), np.load('flabel_57.npy')

x_train, x_test = torch.tensor(x[:12000]), torch.tensor(x[12000:])
y_train, y_test = torch.tensor(y[:12000]), torch.tensor(y[12000:])
```

Both data and label are placed in order of class 0 & 1.  
x_train[:6000]: class 0, x_train[6000:]: class 1  
x_test[:1000]: class 0, y_test[1000:]: class 1  
  
### Ternary (class 5 & 7 & 9)
