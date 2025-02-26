import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFModule(nn.Module):
    def __init__(self, 
                 input_dim,
                 cif_threshold=1.0,
                 cif_conv_channels=256,
                 cif_conv_kernel_size=5, # Librispeech for 5
                 cif_padding=1):
        """初始化 CIF 模組
        
        Args:
            input_dim (int): 輸入特徵的維度
            cif_threshold (float): CIF 整合的閾值，預設為 1.0
            cif_conv_channels (int): CIF 卷積層的通道數，預設為 256
            cif_conv_kernel_size (int): CIF 卷積層的核大小，預設為 3
            cif_padding (int): CIF 卷積層的填充大小，預設為 1
        """
        super().__init__()
        self.cif_threshold = cif_threshold
        
        # CIF 的卷積層，用於處理輸入特徵
        self.cif_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cif_conv_channels,
            kernel_size=cif_conv_kernel_size,
            stride=1,
            padding=cif_padding
        )
        
        # CIF 的線性層，用於生成 alpha 值
        self.layer_norm = nn.LayerNorm(cif_conv_channels)
        self.cif_linear = nn.Linear(cif_conv_channels, 1)
        
    def forward(self, encoder_out, target_length = None):
        """CIF 前向傳播函數
        
        Args:
            encoder_out (torch.Tensor): 編碼器輸出特徵 [B, T, H]
            
        Returns:
            tuple: 
                - integrated_out (torch.Tensor): CIF 處理後的特徵 [B, U, H]
                - alphas (torch.Tensor): 累積的 alpha 值 [B, U]
        """
        batch_size = encoder_out.size(0)
        
        # 計算 alpha 值
        conv_out = self.cif_conv(encoder_out.transpose(1, 2))  # [B, C, T]
        conv_out = conv_out.transpose(1, 2)  # [B, T, C]
        conv_out = self.layer_norm(conv_out)
        conv_out = F.relu(conv_out)
        alpha = self.cif_linear(conv_out)  # [B, T, 1]
        alpha = torch.sigmoid(alpha)  # 轉換到 0~1 範圍 [B, T, 1]
        print("Before Scaling Strategy: ", torch.sum(alpha.squeeze(-1), dim=1))

        # For Quantity Loss
        alphas = [] 
        alphas.append(alpha.sum(dim=1)) 

        if self.training:
            alpha_sum = torch.sum(alpha.squeeze(-1), dim=1)  # [B]
            target_length = target_length.float()
            scale_factor = target_length / alpha_sum
            alpha = alpha * scale_factor.unsqueeze(1).unsqueeze(2)
            print("After Scaling Strategy: ", torch.sum(alpha.squeeze(-1), dim=1))

        # 累積 alpha 值並整合特徵
        integrated_out = []
        current_alpha_sum = 0
        current_feature = 0  # 這裡代表 ha_u

        for t in range(encoder_out.size(1)):
            current_alpha_sum = current_alpha_sum + alpha[:, t]  # αᵃᵤ

            if current_alpha_sum < self.cif_threshold:
                current_feature = current_feature + encoder_out[:, t] * alpha[:, t]  # hᵃᵤ = hᵃᵤ₋₁ + αᵤ * hᵤ
            else:
                alpha_u1 = 1.0 - (current_alpha_sum - alpha[:, t])  # αᵤ₁
                integrated_out.append(current_feature + alpha_u1 * encoder_out[:, t])  # cᵢ = hᵃᵤ₋₁ + αᵤ₁ * hᵤ

                alpha_u2 = alpha[:, t] - alpha_u1
                current_alpha_sum = alpha_u2  # αᵃᵤ = αᵤ₂
                current_feature = alpha_u2 * encoder_out[:, t]  # hᵃᵤ = αᵤ₂ * hᵤ
        if current_alpha_sum > 0: # 解決殘餘部分
            integrated_out.append(current_feature)
                
        integrated_out = torch.stack(integrated_out, dim=1)  # [B, U, H] # torch.stack dim=1 表示水平堆疊, dim=0 表示垂直堆疊
        alphas = torch.stack(alphas, dim=1)  # [B, U]
        
        return integrated_out, alphas

def get_cif_module(input_dim, 
                  cif_threshold=1.0,
                  cif_conv_channels=256,
                  cif_conv_kernel_size=3,
                  cif_padding=1):
    """取得 CIF 模組實例
    
    Args:
        input_dim (int): 輸入特徵的維度
        cif_threshold (float): CIF 整合的閾值，預設為 1.0
        cif_conv_channels (int): CIF 卷積層的通道數，預設為 256
        cif_conv_kernel_size (int): CIF 卷積層的核大小，預設為 3
        cif_padding (int): CIF 卷積層的填充大小，預設為 1
        
    Returns:
        CIFModule: CIF 模組實例
    """
    return CIFModule(
        input_dim=input_dim,
        cif_threshold=cif_threshold,
        cif_conv_channels=cif_conv_channels,
        cif_conv_kernel_size=cif_conv_kernel_size,
        cif_padding=cif_padding
    )