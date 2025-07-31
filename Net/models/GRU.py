from torch import nn
# 定义优化器
import torch


# 搭建GRU网络
class GRUNet(nn.Module):
    def __init__(self,
                 input_size,
                 hid_size,
                 # num_rnn_layers,
                 output_size,
                 dropout_p=0,
                 bidirectional=True,  # 选择使用双向True
                 ):
        super(GRUNet, self).__init__()
        """
        input_size:输入大小/长度
        hidden_dim:GRU神经元个数
        layer_dim:GRU的层数
        output_dim:隐藏层输出的维度（分类的数量）
        dropout_p = 0.2,
        bidirectional = True, #选择使用双向True
        """

        # GRU+全连接层
        # print("GRU_dim",embedding_dim,hidden_dim,layer_dim) # GRU_dim 100 128 1
        # self.hidden_size = hid_size
        # self.gru = nn.GRU(input_size=input_size,
        #                     hidden_size=hid_size,
        #                     num_layers=num_rnn_layers,
        #                     dropout=dropout_p if num_rnn_layers > 1 else 0,
        #                     bidirectional=bidirectional,  # 控制是否使用双向LSTM
        #                     batch_first=True,
        #                     )
        # # print("fc1_dim",hidden_dim,output_dim) # fc1_dim 128 10
        # if bidirectional == True:
        #     self.fc1 = nn.Linear(hid_size * 2, output_size)  # 双向LSTM网络  hid*2
        # else:
        #     self.fc1 = nn.Linear(hid_size, output_size)

        self.dropout = nn.Dropout(dropout_p)
        self.gru_layers = nn.ModuleList()
        self.gru_layers.append(nn.GRU(input_size, hid_size[0], batch_first=True, dropout=dropout_p))
        # self.attn_layers = nn.ModuleList()
        # self.attn_layers.append(nn.MultiheadAttention(hid_size[0], num_heads=4, batch_first=True, bias=True))
        if len(hid_size) > 1:
            for i in range(1, len(hid_size)):
                self.gru_layers.append(nn.GRU(hid_size[i - 1], hid_size[i], batch_first=True, dropout=dropout_p))
                # self.attn_layers.append(nn.MultiheadAttention(hid_size[i], num_heads=4, batch_first=True, bias=True))
        self.fc1 = nn.Linear(hid_size[-1], int(hid_size[-1] / 2))
        self.fc = nn.Linear(int(hid_size[-1] / 2), output_size)

    def forward(self, x):
        for i in range(len(self.gru_layers)):
            out, _ = self.gru_layers[i](x)  # torch.Size([32, 500, 64])
            out = self.dropout(out)
            x = out
        # 取GRU最后一个时间步的输出作为全连接层的输入
        x = x[:, -1, :]
        out = self.fc(self.fc1(x))
        # out = self.fc(x[:, -1, :])
        return out  # 后续交叉熵损失函数自带Softmax层


if __name__ == '__main__':
    print(1)
    import numpy as np
    from scipy import interpolate

    # 原始信号长度为800
    original_length = 8
    original_signal = [1, 2, 3, 4, 4, 3, 2, 1]
    # 目标信号长度为500
    target_length = 5

    # 计算降采样因子
    downsampling_factor = original_length / target_length

    # 创建插值函数
    interp_func = interpolate.interp1d(np.arange(original_length), original_signal, kind='linear')

    # 在新的长度上进行插值
    new_signal = interp_func(np.arange(0, original_length - 1, downsampling_factor))

    print("压缩后的信号:", new_signal)
