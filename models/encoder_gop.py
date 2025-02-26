import torch
from torch import nn
from transformers import AutoModel
#from speechtokenizer import SpeechTokenizer

def get_gop_encoder(name, finetune_encoder, gop_dim, audio_head, audio_depth, audio_hidden_dim):
    if name == "facebook/hubert-xlarge-ll60k":
        return TransformerAudioEnoder(model_name='facebook/hubert-xlarge-ll60k', finetune=finetune_encoder)
    elif name == "conv-branchformer":
        return ConvBrabchformerAudioEnoder(embed_dim=gop_dim, num_heads=audio_head, depth=audio_depth, hidden_dim=audio_hidden_dim)
    elif name == "conv-branchformer-2":
        return ConvBrabchformerAudioEnoder_2(embed_dim=gop_dim, num_heads=audio_head, depth=audio_depth, hidden_dim=audio_hidden_dim)
    else:
        raise NotImplementedError
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        #print(C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ConvBrabchformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.size = dim
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        #[in_channel, out_channel, kernel_size]
        self.conv1d = nn.Conv1d(dim, dim*2, kernel_size=1)
        self.glu_act = nn.GLU()
        self.cnn_norm1 = norm_layer(dim)
        self.cnn_norm2 = norm_layer(dim)
        #depth-wise conv
        self.dep_cnn1 = nn.Conv1d(dim, dim, kernel_size=3, groups=dim, padding=1)
        self.dep_cnn2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.conv1d_2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.cnn_drop = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.2)

        # attention-based pooling for two branches
        self.pooling_proj1 = torch.nn.Linear(dim, 1)
        self.pooling_proj2 = torch.nn.Linear(dim, 1)
        # linear projections for calculating merging weights
        self.weight_proj1 = torch.nn.Linear(dim, 1)
        self.weight_proj2 = torch.nn.Linear(dim, 1)
        # linear projection after weighted average
        self.merge_proj = torch.nn.Linear(dim, dim)

        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # self atten block
        x_att = x + self.drop_path(self.attn(self.norm1(x)))
        x_att = x_att + self.drop_path(self.mlp(self.norm2(x_att)))
        # conv block, #x [B, 50, dim]
        x_cnn = self.conv1d(self.cnn_norm1(x).transpose(1,2)).transpose(1,2) #x_cnn [B, 50, dim*2]
        x_cnn = self.glu_act(x_cnn) #x_cnn [B, 50, dim]
        # depth-wise conv
        x_cnn = self.dep_cnn1(x_cnn.transpose(1,2))
        x_cnn = self.dep_cnn2(x_cnn).transpose(1,2)
        x_cnn = torch.nn.functional.relu(self.cnn_norm2(x_cnn))
        x_cnn = self.cnn_drop(self.conv1d_2(x_cnn.transpose(1,2)).transpose(1,2))
        x_cnn = x + x_cnn 

        # branch1 for atten_out attention pooling
        score1 = (
            self.pooling_proj1(x_att).transpose(1, 2) / self.size**0.5
        )  # (batch, 1, time)
        score1 = torch.softmax(score1, dim=-1)
        pooled1 = torch.matmul(score1, x_att).squeeze(1)  # (batch, size)
        weight1 = self.weight_proj1(pooled1)  # (batch, 1)

        # branch2 for cnn_out attention pooling
        score2 = (
            self.pooling_proj2(x_cnn).transpose(1, 2) / self.size**0.5
        )  # (batch, 1, time)
        score2 = torch.softmax(score2, dim=-1)
        pooled2 = torch.matmul(score2, x_cnn).squeeze(1)  # (batch, size)
        weight2 = self.weight_proj2(pooled2)  # (batch, 1)

        # normalize weights of two branches
        merge_weights = torch.softmax(
            torch.cat([weight1, weight2], dim=-1), dim=-1
        )  # (batch, 2)
        merge_weights = merge_weights.unsqueeze(-1).unsqueeze(-1)  # (batch, 2, 1, 1)
        w1, w2 = merge_weights[:, 0], merge_weights[:, 1]  # (batch, 1, 1)

        # merge and proj
        x = x + self.dropout(
            self.mlp2(w1 * x_att + w2 * x_cnn)
        )
        
        return x

class TransformerAudioEnoder(nn.Module):
    def __init__(self, model_name='facebook/hubert-xlarge-ll60k', finetune=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = finetune
            
        for param in self.encoder.encoder.layers[-15:].parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.encoder(x).last_hidden_state

class ConvBrabchformerAudioEnoder(nn.Module):
    def __init__(self, embed_dim=84, num_heads=3, depth=3, hidden_dim=24):
        super().__init__()

        self.proj = Mlp(in_features=embed_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=0.1)
        self.encoder = nn.ModuleList([ConvBrabchformerBlock(dim=hidden_dim, num_heads=num_heads) for i in range(depth)])

    def forward(self, x):
        x = self.proj(x)
        for blk in self.encoder:
            x = blk(x)
        return x


if __name__ == "__main__":
    model = ConvBrabchformerAudioEnoder(embed_dim=3164, num_heads=1, depth=2)
    # print(model)
    x = torch.randn(1, 22, 3164)
    z = model(x)
    print(z.shape)