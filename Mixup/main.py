# Source: https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py

import numpy as np
import torch.nn as nn
import torch.optim as optim

alpha = 1.0

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)


# (x, y) = a batch of feature and target
# In the original code, y.shape is (batch_size, 1) and its entries are integers between 0 and 9.

lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

batch_size = x.size()[0]

index = torch.randperm(batch_size)

xj, yj = x[index, :], y[index]

x_tilde = lam * x + (1 - lam) * xj

outputs = net(x_tilde)

loss = lam * loss_fn(outputs, y) + (1 - lam) * loss_fn(outputs, yj)

optimizer.zero_grad()
loss.backward()
optimizer.step()
