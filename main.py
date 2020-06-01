import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import os

# 数据格式
transform = transforms.Compose(
						# ToTensor 将输入的数据转换为Tensor的格式
						[transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
						# Normalize([0.5], [0.5] 归一化处理[-1,1]，因为mnist数据集为黑白图片，depth是1，
						# 如果是彩色图片，输入depth，则transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
					)
# 训练集
train_data = datasets.MNIST(
						root='./data/',  # 数据存放的路径
						train=True,  # 作为训练集
						transform=transform,  # 传入数据转化格式
						download=True  # 下载数据集
					)
# 测试集
test_data = torchvision.datasets.MNIST(
						root='./data/',
						train=False,  # 作为测试集
						transform=transform
					)

# 每批装载的数据图片设置为64
Batch_size = 64
# 加载训练数据集
train_data_loader = torch.utils.data.DataLoader(
						dataset=train_data,  # 数据集为训练集
						batch_size=Batch_size,
						shuffle=True  # 随机加载
					)
# 加载测试数据集
test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=Batch_size, shuffle=False)

# 构建神经网络模型
class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		# 卷积层
		self.conv = torch.nn.Sequential(
			# 输入是1，输出是16，卷积核是3*3，每次移动1步，每一条边补充1行/列的0，
			# 经过这一步之后，数据由1*28*28，变成了16*28*28
			torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
			# 激活函数
			torch.nn.ReLU(),
			# 输入是16，输出是32，卷积核是3*3，每次移动1步，每一条边补充1行/列的0，
			# 经过这一步之后，数据由16*28*28，变成了32*28*28
			torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
			# 激活函数
			torch.nn.ReLU(),
			# 池化层
			# 每次移动2步，卷积核是2*2，即每2*2的矩阵中的4个数据选取一个最大值，
			# 这样就由32*28*28，变成了32*14*14
			torch.nn.MaxPool2d(stride=2, kernel_size=2)
		)

		# 全连接层
		self.dense = torch.nn.Sequential(
			# 将32*14*14的数据线性转化为1024的数据
			torch.nn.Linear(32 * 14 * 14, 1024),
			# 激活函数
			torch.nn.ReLU(),
			# 设置丢弃的概率，防止模型的过拟合
			torch.nn.Dropout(p=0.3),
			# 将1024的数据线性转化为10的数据，即0-9
			torch.nn.Linear(1024, 10)
		)

	# 向前传播
	def forward(self, x):
		# 卷积
		x = self.conv(x)
		# 扁平化处理
		x = x.view(-1, 32 * 14 * 14)
		# 全连接
		x = self.dense(x)
		return x

# 构建神经网络
model = Model()
print(model)
print()

if os.path.exists("model.pkl"):
	print("导入已训练好的模型")
	model.load_state_dict(torch.load('model.pkl'))
else:
	# 启用BatchNormalization和Dropout，修改权值
	model.train()
	# 定义优化器
	optimizer = torch.optim.Adam(model.parameters())
	# 定义损失函数
	loss_func = torch.nn.CrossEntropyLoss()
	# 设置迭代次数为5次
	n_epochs = 5
	# 训练模型
	for epoch in range(n_epochs):
		train_loss = 0.  # 训练的损失值
		train_correct = 0.  # 训练正确的个数
		print("Epoch {}/{}".format(epoch, n_epochs))
		print("--------------------------------------------")
		for img, label in train_data_loader:
			# 图片和标签
			img, label = Variable(img), Variable(label)
			# 得到结果，每个结果为一行，一行10个值，表示0-9的概率
			outputs = model(img)
			# 获取1维度即行维度（每行）的最大值（_）和最大值对应的索引（pred）
			_, pred = torch.max(outputs.data, 1)
			# 求loss
			loss = loss_func(outputs, label)

			# 清零，清除上一次结果的影响
			optimizer.zero_grad()
			# 反向传播
			loss.backward()
			# 更新所有的参数、优化
			optimizer.step()

			# .item()获取最里面的值，此处即loss值
			train_loss += loss.item()
			# 统计每批数据的正确个数
			train_correct += torch.sum(pred == label.data)

		# 平均损失值
		train_loss_avg = train_loss / len(train_data)
		# 正确率
		train_acc = 100 * train_correct / len(train_data)
		print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%".format(train_loss_avg, train_acc))
		print("--------------------------------------------")
		print()

# 测试
model.eval()  # 仅测试，不修改权值
test_correct = 0.  # 保存正确的个数
for img, label in test_data_loader:
	# 图片和标签
	img, label = Variable(img), Variable(label)
	# 得到结果，每个结果为一行，一行10个值，表示0-9的概率
	outputs = model(img)
	# 获取1维度即行维度（每行）的最大值（_）和最大值对应的索引（pred）
	_, pred = torch.max(outputs.data, 1)
	# 统计每批数据的正确个数
	test_correct += torch.sum(pred == label.data)

# 正确率
test_acc = 100 * test_correct / len(test_data)
print("Test Accuracy is:{:.4f}%".format(test_acc))

# 保存训练好的模型
torch.save(model.state_dict(), "model.pkl")