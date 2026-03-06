import torch
import torch.nn as nn

class SOH_Gated_GRU(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super(SOH_Gated_GRU, self).__init__()
        
        # 1. 主干网络 (处理 I, V, T)
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.1
        )
        
        # 2. 注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # 3. SOH 门控模块 (核心创新)
        # 将 SOH (标量) 映射为 hidden_size 维度的调节向量
        self.soh_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_size),
            nn.Sigmoid() # 输出 0~1 的权重
        )
        
        # 4. 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # 输出 SOC
        )

    def forward(self, x, soh=None):
        """
        x: [Batch, Seq_Len, 3] -> (Current, Voltage, Temperature)
        soh: [Batch, 1]
        """
        # GRU 特征提取
        gru_out, _ = self.gru(x)
        
        # Attention 加权
        weights = self.attention(gru_out) 
        context = torch.sum(weights * gru_out, dim=1) # [Batch, Hidden]
        
        # SOH 门控调节
        if soh is not None:
            gate_factor = self.soh_gate(soh)
            # 门控融合：利用 SOH 调整特征强度
            # 物理含义：不同 SOH 下，同样的电压电流意味着不同的 SOC
            features_refined = context * (1 + gate_factor)
        else:
            features_refined = context
            
        return self.fc(features_refined)