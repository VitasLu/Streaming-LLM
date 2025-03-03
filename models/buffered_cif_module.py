import torch
import torch.nn as nn
import torch.nn.functional as F

class BufferedCIFModule(nn.Module):
    def __init__(self, 
                input_dim,
                cif_threshold=1.0,
                cif_conv_channels=256,
                cif_conv_kernel_size=5,
                cif_padding=2):
        """CIF 模組 (含緩衝區)
        
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
        
        # 緩衝區狀態 - 使用字典保存每個音頻ID的狀態
        self.buffers = {}
        
    def reset_buffer(self, audio_id=None):
        """重置緩衝區狀態
        
        Args:
            audio_id (str, optional): 指定重置的音頻ID，None表示重置所有
        """
        if audio_id is None:
            # 重置所有緩衝區
            self.buffers = {}
        else:
            # 重置特定音頻ID的緩衝區
            self.buffers[audio_id] = {
                'alpha_sum': 0,
                'feature': 0
            }
    
    def clear_audio_buffer(self, audio_id):
        """移除特定音頻ID的緩衝區
        
        Args:
            audio_id (str): 要移除的音頻ID
        """
        if audio_id in self.buffers:
            del self.buffers[audio_id]
        
    def forward(self, encoder_out, audio_id=None, target_length=None, is_final_segment=False):
        """CIF 前向傳播函數 (含緩衝區支持)
        
        Args:
            encoder_out (torch.Tensor): 編碼器輸出特徵 [B, T, H]
            audio_id (str, optional): 音頻ID，用於跟踪狀態
            target_length (torch.Tensor, optional): 目標長度 (訓練時使用)
            is_final_segment (bool): 是否為最後一個段落，用於處理殘餘特徵
            
        Returns:
            tuple: 
                - integrated_out (torch.Tensor): CIF 處理後的特徵 [B, U, H]
                - alphas (torch.Tensor): 累積的 alpha 值 [B, U]
        """
        batch_size = encoder_out.size(0)
        
        # 確保音頻ID的狀態存在
        if audio_id is not None and audio_id not in self.buffers:
            self.reset_buffer(audio_id)
        
        # 計算 alpha 值
        conv_out = self.cif_conv(encoder_out.transpose(1, 2))  # [B, C, T]
        conv_out = conv_out.transpose(1, 2)  # [B, T, C]
        conv_out = self.layer_norm(conv_out)
        conv_out = F.relu(conv_out)
        alpha = self.cif_linear(conv_out)  # [B, T, 1]
        alpha = torch.sigmoid(alpha)  # 轉換到 0~1 範圍 [B, T, 1]

        # For Quantity Loss
        alphas = [] 
        alphas.append(alpha.sum(dim=1)) 

        if self.training and target_length is not None:
            alpha_sum = torch.sum(alpha.squeeze(-1), dim=1)  # [B]
            target_length = target_length.float()
            scale_factor = target_length / alpha_sum
            alpha = alpha * scale_factor.unsqueeze(1).unsqueeze(2)

        # 累積 alpha 值並整合特徵
        integrated_out = []
        
        # 獲取或初始化狀態
        if audio_id is not None:
            # 使用特定音頻ID的狀態
            current_state = self.buffers[audio_id]
            current_alpha_sum = current_state['alpha_sum']
            current_feature = current_state['feature']
        else:
            # 臨時狀態 (不保存)
            current_alpha_sum = 0
            current_feature = 0
        
        for t in range(encoder_out.size(1)):
            current_alpha_sum = current_alpha_sum + alpha[:, t]  # αᵃᵤ
            
            if current_alpha_sum < self.cif_threshold:
                if isinstance(current_feature, int) and current_feature == 0:
                    current_feature = encoder_out[:, t] * alpha[:, t]  # 初始化特徵
                else:
                    current_feature = current_feature + encoder_out[:, t] * alpha[:, t]  # hᵃᵤ = hᵃᵤ₋₁ + αᵤ * hᵤ
            else:
                # 如果超過閾值，產生一個輸出項並重置累積
                alpha_u1 = 1.0 - (current_alpha_sum - alpha[:, t])  # αᵤ₁
                
                # 確保 current_feature 是有效的張量
                if isinstance(current_feature, int) and current_feature == 0:
                    current_feature = torch.zeros_like(encoder_out[:, 0])
                    
                integrated_out.append(current_feature + alpha_u1 * encoder_out[:, t])  # cᵢ = hᵃᵤ₋₁ + αᵤ₁ * hᵤ

                # 更新剩餘值
                alpha_u2 = alpha[:, t] - alpha_u1
                current_alpha_sum = alpha_u2  # αᵃᵤ = αᵤ₂
                current_feature = alpha_u2 * encoder_out[:, t]  # hᵃᵤ = αᵤ₂ * hᵤ
        
        # 如果是最後一個段落，輸出剩餘特徵
        if is_final_segment and current_alpha_sum > 0:
            if isinstance(current_feature, int) and current_feature == 0:
                current_feature = torch.zeros_like(encoder_out[:, 0])
            integrated_out.append(current_feature)
            current_alpha_sum = 0
            current_feature = 0
        
        # 更新狀態
        if audio_id is not None:
            self.buffers[audio_id]['alpha_sum'] = current_alpha_sum
            self.buffers[audio_id]['feature'] = current_feature
            
        # 若無輸出，返回None
        if len(integrated_out) == 0:
            return None, torch.stack(alphas, dim=1)
            
        integrated_out = torch.stack(integrated_out, dim=1)  # [B, U, H]
        alphas = torch.stack(alphas, dim=1)  # [B, 1]
        
        return integrated_out, alphas

def get_buffered_cif_module(input_dim, 
                           cif_threshold=1.0,
                           cif_conv_channels=256,
                           cif_conv_kernel_size=3,
                           cif_padding=1):
    """取得 CIF 模組實例 (含緩衝區支持)
    
    Args:
        input_dim (int): 輸入特徵的維度
        cif_threshold (float): CIF 整合的閾值，預設為 1.0
        cif_conv_channels (int): CIF 卷積層的通道數，預設為 256
        cif_conv_kernel_size (int): CIF 卷積層的核大小，預設為 3
        cif_padding (int): CIF 卷積層的填充大小，預設為 1
        
    Returns:
        BufferedCIFModule: CIF 模組實例
    """
    return BufferedCIFModule(
        input_dim=input_dim,
        cif_threshold=cif_threshold,
        cif_conv_channels=cif_conv_channels,
        cif_conv_kernel_size=cif_conv_kernel_size,
        cif_padding=cif_padding
    )