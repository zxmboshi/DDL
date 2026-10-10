import torch
import torch.nn as nn
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self,dim,num_heads,dim_head):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        weight_dim = num_heads * dim_head
        self.qkv = nn.Linear(dim, weight_dim*3)
        self.proj = nn.Linear(weight_dim,dim)

    def forward(self,x):
        qkv = self.qkv(x).chunk(3,dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(dots,dim=-1)
        out = torch.matmul(attn,v)
        out = rearrange(out,'b h n d -> b n (h d)')
        out = self.proj(out)
        return out


class FFN(nn.Module):
    def __init__(self,dim,hidden_dim,drop_out=0.):
        super(FFN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim,dim),
            nn.GELU(),
            nn.Dropout(drop_out)
        )

    def forward(self,x):
        out = self.net(x)
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Transformer(nn.Module):
    def __init__(self, dim, num_heads, depth, embed_dim, mlp_dim, dropout=0.1):
        super().__init__()
        dim_head = embed_dim // num_heads
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadAttention(dim, num_heads, dim_head)),
                PreNorm(dim, FFN(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class BatchFormer(nn.Module):
    def __init__(self,in_channels, layers, mlp_dim,embed_dim=96,num_heads=4):
        super(BatchFormer, self).__init__()
        self.transformer = Transformer(in_channels, num_heads, layers, embed_dim//num_heads, mlp_dim=mlp_dim, dropout=0.1)

    def forward(self,x):
        orig_x, x = torch.split(x, len(x) // 2)
        x = self.transformer(x)
        x = torch.cat([orig_x, x], dim=0)
        return x


if __name__ == '__main__':
    # n, b, c
    x = torch.randn(1000,4,256)
    # x = torch.flatten(x,1).unsqueeze(1)
    model = BatchFormer(256,2,256,1024)
    y = model(x)
    print(y.shape)


