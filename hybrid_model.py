import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    通用 Patch Embedding
    Input: (Batch, Seq_Len, in_channels)
    Output: (Batch, Num_Patches, D_Model)
    """

    def __init__(self, d_model, patch_len, stride, dropout, in_channels=1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

        # 线性投影层：将展平的 Patch 映射到 Latent Space
        self.proj = nn.Linear(patch_len * in_channels, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, in_channels]
        seq_len = x.shape[1]

        # 计算 Patch 数量
        num_patches = (seq_len - self.patch_len) // self.stride + 1

        patches = []
        for i in range(num_patches):
            start = i * self.stride
            end = start + self.patch_len
            patches.append(x[:, start:end, :])

        # 堆叠 -> [Batch, Num_Patches, Patch_Len, C]
        x = torch.stack(patches, dim=1)
        # 展平 Patch 维度 -> [Batch, Num_Patches, Patch_Len*C]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # 投影 -> [Batch, Num_Patches, D_Model]
        x = self.proj(x)
        return self.dropout(x)


class DualStreamPatchTST(nn.Module):
    """
    基于模态聚类的双流解耦预测模型 (Hybrid Model)
    包含三个阶段：
    1. 形态感知模态识别 (在数据预处理阶段通过聚类完成，此处接收特定模态数据)
    2. 双流解耦特征提取 (Load Stream & Temp Stream)
    3. 热动力学引导融合 (Thermodynamic Guided Fusion)
    """

    def __init__(self, configs, num_transformers):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_transformers = num_transformers

        # 参数获取
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)
        self.d_model = getattr(configs, 'd_model', 128)
        self.n_heads = getattr(configs, 'n_heads', 4)
        self.e_layers = getattr(configs, 'e_layers', 2)
        self.d_ff = getattr(configs, 'd_ff', 256)
        self.dropout = getattr(configs, 'dropout', 0.1)

        # =========================================================
        # 1. 输入层：双流 Patch Embedding
        # =========================================================
        # 通道 1: 处理负载 (Load) - 高频特征
        self.load_embedding = PatchEmbedding(
            d_model=self.d_model, patch_len=self.patch_len, stride=self.stride,
            dropout=self.dropout, in_channels=1
        )
        # 通道 2: 处理温度 (Temp) - 低频特征
        self.temp_embedding = PatchEmbedding(
            d_model=self.d_model, patch_len=self.patch_len, stride=self.stride,
            dropout=self.dropout, in_channels=1
        )

        # =========================================================
        # 2. 编码层：双塔 Transformer Encoder
        # =========================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.d_ff,
            dropout=self.dropout, activation='gelu', batch_first=True, norm_first=True
        )

        # 负载编码器
        self.load_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.e_layers)
        # 温度编码器
        self.temp_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.e_layers)

        # =========================================================
        # 3. 融合层：热动力学引导门控 (Thermodynamic Gating)
        # =========================================================
        # 门控生成层：将温度特征映射为 0-1 的门控信号
        self.thermo_gate_proj = nn.Linear(self.d_model, self.d_model)

        # 温度投影层：用于将温度特征对齐并添加到最终结果中 (Residual Path)
        self.temp_proj = nn.Linear(self.d_model, self.d_model)

        # =========================================================
        # 4. 输出层：预测头
        # =========================================================
        # 计算经过 Patching 后的序列长度
        num_patches = (self.seq_len - self.patch_len) // self.stride + 1
        self.head = nn.Linear(num_patches * self.d_model, self.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        x_enc shape: (Batch, Seq_Len, 2 * num_transformers)
        数据排列假设：前 num_transformers 列是 Load，后 num_transformers 列是 Temp
        """
        batch_size = x_enc.shape[0]

        # ---------------------------------------------------------
        # Step 1: 数据解耦与重塑 (Data Decoupling & Reshaping)
        # ---------------------------------------------------------
        # 分离负载和温度
        # x_load: (Batch, Seq, N) -> (Batch, N, Seq)
        x_load = x_enc[:, :, :self.num_transformers].permute(0, 2, 1)
        # x_temp: (Batch, Seq, N) -> (Batch, N, Seq)
        x_temp = x_enc[:, :, self.num_transformers:].permute(0, 2, 1)

        # 展平 Batch 和 Transformer 维度，实现“通道独立 (Channel Independence)”处理
        # Shape: (Batch * N, Seq, 1)
        x_load = x_load.reshape(batch_size * self.num_transformers, self.seq_len, 1)
        x_temp = x_temp.reshape(batch_size * self.num_transformers, self.seq_len, 1)

        # ---------------------------------------------------------
        # Step 2: 双流特征提取 (Dual-Stream Encoding)
        # ---------------------------------------------------------
        # Load Stream (High Frequency)
        load_emb = self.load_embedding(x_load)  # (B*N, Num_Patches, D)
        load_out = self.load_encoder(load_emb)  # (B*N, Num_Patches, D)

        # Temp Stream (Low Frequency)
        temp_emb = self.temp_embedding(x_temp)  # (B*N, Num_Patches, D)
        temp_out = self.temp_encoder(temp_emb)  # (B*N, Num_Patches, D)

        # ---------------------------------------------------------
        # Step 3: 热动力学引导融合 (Thermodynamic Guided Fusion)
        # ---------------------------------------------------------
        # (1) 生成热动力学门控 (Gate)
        # Sigmoid 确保值在 (0, 1) 之间，表示“激活概率”或“通过率”
        gate = torch.sigmoid(self.thermo_gate_proj(temp_out))

        # (2) 门控机制作用于负载 (Gating Action)
        # 物理含义：温度条件决定了多少历史负载特征是有效的
        guided_load = load_out * gate

        # (3) 特征融合 (Fusion with Residual)
        # 融合特征 = 门控后的负载 + 变换后的温度上下文
        fused_out = guided_load + self.temp_proj(temp_out)

        # ---------------------------------------------------------
        # Step 4: 预测输出 (Prediction)
        # ---------------------------------------------------------
        # 展平 Patch 维度 -> (B*N, Num_Patches * D)
        fused_out = fused_out.reshape(fused_out.shape[0], -1)

        # 线性映射到预测长度 -> (B*N, Pred_Len)
        output = self.head(fused_out)

        # 还原维度 -> (Batch, N, Pred_Len) -> (Batch, Pred_Len, N)
        output = output.reshape(batch_size, self.num_transformers, self.pred_len).permute(0, 2, 1)

        return output