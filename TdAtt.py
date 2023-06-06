import math
import torch
from torch import nn
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None):
    # input is 4 dimension tensor
    # [batch_size, head, length, d_tensor]
    d_k = q.size()[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None:
        #mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        
    scores = F.softmax(scores, dim=-1)
    output = torch.matmul(scores, v)
    return output, scores

class TwoD_Attention_layer(nn.Module):
    def __init__(self, in_channels=64, num_head=64, emb_dim=64, layernorm_eps=1e-6): #n
        super(TwoD_Attention_layer, self).__init__()
        self.in_channels = in_channels
        self.num_head = num_head #c
        self.emb_dim = emb_dim
        self.convq = nn.Conv2d(self.in_channels, self.num_head, kernel_size=3, stride=1, padding='same')
        self.convk = nn.Conv2d(self.in_channels, self.num_head, kernel_size=3, stride=1, padding='same')
        self.convv = nn.Conv2d(self.in_channels, self.num_head, kernel_size=3, stride=1, padding='same')
        self.conv = nn.Conv2d(self.num_head * 2, self.in_channels, kernel_size=3, stride=1, padding='same') #n
        self.bnq = nn.BatchNorm2d(self.num_head)
        self.bnk = nn.BatchNorm2d(self.num_head)
        self.bnv = nn.BatchNorm2d(self.num_head)
        self.ln = nn.LayerNorm(self.emb_dim, eps=layernorm_eps) # normalized_shape=embedding_dim
        
        self.final_conv1 = nn.Conv2d(self.in_channels, self.in_channels, 3, padding='same') # activation='relu'
        self.act1 = nn.ReLU()
        self.final_conv2 = nn.Conv2d(self.in_channels, self.in_channels, 3, padding='same')
        
        self.bnf1 = nn.BatchNorm2d(self.in_channels)
        self.bnf2 = nn.BatchNorm2d(self.in_channels)
        
        self.act2 = nn.ReLU()
        
    def forward(self, inputs):
        residual = inputs
        batch_size = inputs.shape[0]
        
        q_time = self.bnq(self.convq(inputs))
        k_time = self.bnk(self.convk(inputs))
        v_time = self.bnv(self.convv(inputs))
        
        q_fre = torch.permute(q_time, (0, 1, 3, 2))
        k_fre = torch.permute(k_time, (0, 1, 3, 2))
        v_fre = torch.permute(v_time, (0, 1, 3, 2))
        
        scaled_attention_time, attention_weights_time = scaled_dot_product_attention(q_time, k_time, v_time, None)
        scaled_attention_fre, attention_weights_fre = scaled_dot_product_attention(q_fre, k_fre, v_fre, None)
        
        scaled_attention_fre = torch.permute(scaled_attention_fre, (0,1,3,2))
        
        out = torch.cat((scaled_attention_time,scaled_attention_fre), 1) 
        out = self.ln(self.conv(out) + residual)
        
        final_out = self.act1(self.bnf1(self.final_conv1(out)))
        final_out = self.bnf2(self.final_conv2(final_out))

        final_out = self.act2(final_out + out)

        return final_out