import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2TFEquivalent(nn.Module):
    """改进版 CNN2：
    - 保持与原 TF 模型结构对应（两层 Conv 后接全连接）
    - 将卷积层 dropout 与全连接层 dropout 分离（卷积层缺省 0）
    - 可选 Xavier 初始化（更接近 Keras glorot_uniform）
    输入: [B,1,2,128]
    """
    def __init__(self, num_classes: int, dropout_p: float = 0.5, conv_dropout: float = 0.0, use_xavier: bool = True):
        super().__init__()
        self.pad1 = nn.ZeroPad2d((2,2,0,0))
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1,3), padding=0)
        self.pad2 = nn.ZeroPad2d((2,2,0,0))
        self.conv2 = nn.Conv2d(256, 80, kernel_size=(2,3), padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_conv = nn.Dropout(p=conv_dropout) if conv_dropout > 0 else nn.Identity()
        self.dropout_fc = nn.Dropout(p=dropout_p)

        with torch.no_grad():
            x = torch.zeros(1, 1, 2, 128)
            x = self.pad1(x)
            x = self.relu(self.conv1(x))   # [1,256,2,130]
            x = self.dropout_conv(x)
            x = self.pad2(x)
            x = self.relu(self.conv2(x))   # [1,80,1,132]
            x = self.dropout_conv(x)
            flat = x.numel()
        self.fc1 = nn.Linear(flat, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_out = nn.Linear(256, num_classes)

        if use_xavier:
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pad1(x)
        x = self.relu(self.conv1(x))
        x = self.dropout_conv(x)
        x = self.pad2(x)
        x = self.relu(self.conv2(x))
        x = self.dropout_conv(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout_fc(x)
        return self.fc_out(x)
