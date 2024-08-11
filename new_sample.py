import torch
from torch import nn

# 定义模型
vocab_size = 28
embedding_dim = 128
hidden_dim = 256
num_layers = 3
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



# 读取模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProteinGenerator(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)

# 加载模型
state_dict = torch.load('protein_generator_model_final.pth', map_location=device)

# 手动调整权重
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('fc.1'):
        new_key = key.replace('fc.1', 'fc')
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

# 加载调整后的权重
model.load_state_dict(new_state_dict)

model.eval()

print("Model layers dimensions:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")



# 定义字符到索引和索引到字符的映射
char_to_idx = {'1': 1, '2': 2, 'A': 3, 'B': 4, 'C': 5, 'D': 6, 'E': 7, 'F': 8, 'G': 9, 'H': 10, 'I': 11, 'J': 12, 'K': 13, 'L': 14, 'M': 15, 'N': 16, 'O': 17, 'P': 18, 'Q': 19, 'R': 20, 'S': 21, 'T': 22, 'U': 23, 'V': 24, 'W': 25, 'X': 26, 'Y': 27, 'Z': 28}
idx_to_char = {
    1: '1',
    2: '2',
    3: 'A',
    4: 'B',
    5: 'C',
    6: 'D',
    7: 'E',
    8: 'F',
    9: 'G',
    10: 'H',
    11: 'I',
    12: 'J',
    13: 'K',
    14: 'L',
    15: 'M',
    16: 'N',
    17: 'O',
    18: 'P',
    19: 'Q',
    20: 'R',
    21: 'S',
    22: 'T',
    23: 'U',
    24: 'V',
    25: 'W',
    26: 'X',
    27: 'Y',
    28: 'Z'
}

# 提供一个小部分的蛋白质序列开头
seed_sequence = "KEQAEGEAVTATATPVV"
print("Seed sequence:", seed_sequence)

# 将种子序列转换为索引序列
indexed_seed_sequence = [char_to_idx[char] for char in seed_sequence]

# 设置最大序列长度
maxlen = 100

# 填充序列
padded_seed_sequence = indexed_seed_sequence + [0] * (maxlen - len(indexed_seed_sequence))
assert len(padded_seed_sequence) == maxlen, f"Expected length {maxlen}, but got {len(padded_seed_sequence)}"

# 转换为张量
input_tensor = torch.tensor([padded_seed_sequence], dtype=torch.long).to(device)
assert input_tensor.shape == (1, maxlen), f"Expected shape (1, {maxlen}), but got {input_tensor.shape}"
# 生成蛋白质序列
generated_sequence = []
with torch.no_grad():
    for _ in range(maxlen):  # 生成与 maxlen 长度相同的序列
        output = model(input_tensor)
        probs = torch.softmax(output.reshape(-1), dim=0)
        next_char_idx = torch.multinomial(probs, num_samples=1).item()

        while next_char_idx in [1, 2] or next_char_idx >= vocab_size:
            if next_char_idx in [ 1, 2]:
                print(f"Invalid index generated: {next_char_idx}, regenerating...")
                next_char_idx = torch.multinomial(probs, num_samples=1).item()
            else:
                next_char_idx = (next_char_idx-1) % 28

        generated_sequence.append(idx_to_char[next_char_idx])
        input_tensor = torch.cat([input_tensor[:, 1:], torch.tensor([[next_char_idx]], dtype=torch.long).to(device)], dim=1)

generated_protein_sequence = ''.join(generated_sequence)
print("Generated Protein Sequence:",seed_sequence+generated_protein_sequence)