import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter

# 假设你的数据文件名为 'protein_sequences.txt'
data_file = 'data.txt'

# 读取数据并进行预处理
sequences = []
with open(data_file, 'r') as f:
    for line in f:
        sequences.append(line.strip())

# 计算词汇表大小
vocab = {'1','2','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'}
vocab_size = len(vocab)
print('词汇表大小:', vocab_size)
Counter=Counter()
for seq in sequences:
    for char in seq:
        Counter[char]+=1
print(Counter)
# 创建字符到索引的映射
char_to_idx = {char: idx for idx, char in enumerate(vocab)}

# 将序列转换为索引序列
indexed_sequences = [[char_to_idx[char] for char in seq] for seq in sequences]

# 设置最大序列长度
maxlen = max(len(seq) for seq in indexed_sequences)
print(maxlen)

# 填充序列
padded_sequences = [seq + [0] * (maxlen - len(seq)) for seq in indexed_sequences]

# 转换为张量
X = torch.tensor(padded_sequences, dtype=torch.long)

# 定义数据集类
class ProteinDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

# 创建数据加载器
dataset = ProteinDataset(X)
dataloader = DataLoader(dataset, batch_size=48, shuffle=True)

# 定义模型
class ProteinGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(ProteinGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# 实例化模型
embedding_dim = 128
hidden_dim = 256
num_layers = 3  # 增加 LSTM 层的数量
model = ProteinGenerator(vocab_size, embedding_dim, hidden_dim, num_layers)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 添加 dropout 层以防止过拟合
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    model.fc
)

# 将模型和数据移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('使用设备:', device)
model.to(device)

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')):
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs.reshape(-1, vocab_size), batch.reshape(-1))
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    # 每训练完一轮保存一次模型
    torch.save(model.state_dict(), f'protein_generator_model_epoch_{epoch + 1}.pth')

# 保存最终模型
torch.save(model.state_dict(), 'protein_generator_model_final.pth')