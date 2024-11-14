import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as opt
from paddle.vision import transforms
from paddle.vision.datasets import ImageFolder,DatasetFolder
from paddle.io import DataLoader
import argparse
from pprint import pprint
import pickle
from paddle.vision.models import mobilenet_v1


parser=argparse.ArgumentParser("classification")
parser.add_argument('--pathtrain',default=r'../CatDogDataSet/train',help='train data')
parser.add_argument('--pathval',default=r'../CatDogDataSet/val',help='val data')
parser.add_argument('--bs',default=4,type=int,help='batchsize')
parser.add_argument('--lr',default=0.0002)
parser.add_argument('--epochs',default=100)
parser.add_argument('--size',default=(224,224),help='(H,W)')
args=parser.parse_args()

data_transform_train = transforms.Compose([
    transforms.Resize(args.size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(prob=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_transform_val = transforms.Compose([
    transforms.Resize(args.size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = DatasetFolder(root=args.pathtrain, transform=data_transform_train)
val_dataset = DatasetFolder(root=args.pathval, transform=data_transform_val)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs,shuffle=True,drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.bs)

num_classes=len(train_dataset.classes)
total_samples=len(train_dataset.samples)
total_val_sample=len(val_dataset.samples)

#保存标签到pickle
with open('./lable.plk','wb') as f:
    pickle.dump(train_dataset.class_to_idx,f)

net=mobilenet_v1(num_classes=num_classes,pretrained=False)


# 设置优化器
optim = paddle.optimizer.Adam(parameters=net.parameters())
# 设置损失函数
loss_fn = paddle.nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    net.train()
    for batch_id, data in enumerate(train_loader()):
        x_data = data[0]  # 训练数据
        y_data = data[1]  # 训练数据标签
        predicts =net(x_data)  # 预测结果

        # 计算损失 等价于 prepare 中loss的设置
        loss = loss_fn(predicts, y_data)

        # 计算准确率 等价于 prepare 中metrics的设置
        predicted = paddle.argmax(predicts, axis=1)
        y_data=paddle.to_tensor(y_data,dtype='int64')
        batch_correct_imgs =paddle.equal(predicted,y_data).sum()

        # 下面的反向传播、打印训练信息、更新参数、梯度清零都被封装到 Model.fit() 中
        # 反向传播
        loss.backward()

        if (batch_id + 1) % 2 == 0:
            print(
                "epoch: {}, batch_id: {}, loss is: {:.4f}, acc is: {:.4f}".format(
                    epoch, batch_id + 1, loss.numpy(), batch_correct_imgs.numpy()/args.bs
                )
            )
        # 更新参数
        optim.step()
        # 梯度清零
        optim.clear_grad()

    net.eval()
    total_right_sample=0
    for batch_id, data in enumerate(val_loader()):
        x_data = data[0]  # 训练数据
        y_data = data[1]  # 训练数据标签
        predicts =net(x_data)  # 预测结果

        # 计算损失 等价于 prepare 中loss的设置
        loss = loss_fn(predicts, y_data)

        # 计算准确率 等价于 prepare 中metrics的设置
        predicted = paddle.argmax(predicts, axis=1)
        y_data=paddle.to_tensor(y_data,dtype='int64')
        batch_correct_imgs =paddle.equal(predicted,y_data).sum()
        total_right_sample+=batch_correct_imgs.numpy()
    print('acc:{:.2f}%'.format(total_right_sample/total_val_sample*100))
    paddle.save(net.state_dict(), "dynamics_model/model.pdparams")
    paddle.save(optim.state_dict(), "dynamics_model/model.pdopt")