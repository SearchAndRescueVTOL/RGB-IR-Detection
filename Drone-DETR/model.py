from torchvision import transforms
import torch.nn.functional as F
from Conv import Conv, DCNv2
from Blocks import C2f, Partial_conv3
import cv2
import torch
import torch.nn as nn
import math
import time
BATCH_SIZE = 1
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="gelu",attn_dropout=None, act_dropout=None, normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, pos_embed=None, query_pos_embed=None):
        tgt_mask = torch.tensor(tgt_mask, dtype=torch.bool) if tgt_mask is not None else None

        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)[0]
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        q = self.with_pos_embed(tgt, query_pos_embed)
        k = self.with_pos_embed(memory, pos_embed)
        tgt = self.cross_attn(q, k, value=memory, attn_mask=memory_mask)[0]
        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, pos_embed=None, query_pos_embed=None):
        tgt_mask = torch.tensor(tgt_mask, dtype=torch.bool) if tgt_mask is not None else None

        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           pos_embed=pos_embed, query_pos_embed=query_pos_embed)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class RTDETRDecoder(nn.Module):
    def __init__(self, num_classes):
        super(RTDETRDecoder, self).__init__()
        self.num_classes = num_classes
        d_model = 512
        nhead = 8
        dim_feedforward = 1024
        dropout = 0.3 
        activation = "gelu"

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None, pos_embed=None, query_pos_embed=None):
        output = self.transformer_decoder(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, pos_embed=pos_embed, query_pos_embed=query_pos_embed)
        return output


class FFLayer(nn.Module):
    def __init__(self, embedd_dim,ffn_dim,dropout,activation):
        super().__init__()
        self.feed_forward = nn.Sequential(
                nn.Linear(embedd_dim, ffn_dim),
                getattr(nn, activation)(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embedd_dim)
            )

        self.norm = nn.LayerNorm(embedd_dim)    
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        residual = x
        x = self.feed_forward(x)
        out = self.norm(residual + x)
        return out

class TransformerEncoderLayer(nn.Module):
    
    def __init__(self,hidden_dim,n_head,ffn_dim:int,dropout:float,activation:str = "ReLU"):  
            
        super().__init__()    
        self.mh_self_attn = nn.MultiheadAttention(hidden_dim, n_head, dropout, batch_first=True)
        self.feed_foward_nn = FFLayer(hidden_dim,ffn_dim,dropout,activation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self,x:torch.Tensor,mask:torch.Tensor=None,pos_emb:torch.Tensor=None) -> torch.Tensor:
        residual = x
        q = k = x + pos_emb if pos_emb is not None else x
        x, attn_score = self.mh_self_attn(q, k, value=x, attn_mask=mask)
        x= residual + self.dropout(x)
        x = self.norm(x)
         
        x = self.feed_foward_nn(x)

        return x
        
class AIFI(nn.Module):
  
  def __init__(self):
      super().__init__()
      self.hidden_dim = 256
      self.num_layers = 1
      self.num_heads = 8
      self.dropout = 0.2
      self.eval_spatial_size = [640,640]
      self.pe_temperature = 10000
      self.projection = nn.Linear(256, self.hidden_dim)
      pos_embed = self.build_2d_sincos_position_embedding(
          self.eval_spatial_size[1] // 32, self.eval_spatial_size[0] // 32,
          self.hidden_dim, self.pe_temperature)
      setattr(self, 'pos_embed', pos_embed)
      self.transformer_encoder= TransformerEncoderLayer(
          hidden_dim = self.hidden_dim,
          n_head = self.num_heads,
          ffn_dim = 1024,
          dropout = self.dropout,
          activation = 'GELU'
      )
  
  @staticmethod
  def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
      '''
      source code from: https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/src/zoo/rtdetr/hybrid_encoder.py
      '''

      grid_w = torch.arange(int(w), dtype=torch.float32)
      grid_h = torch.arange(int(h), dtype=torch.float32)
      grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
      assert embed_dim % 4 == 0, \
          'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
      pos_dim = embed_dim // 4
      omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
      omega = 1. / (temperature ** omega)
      out_w = grid_w.flatten()[..., None] @ omega[None]
      out_h = grid_h.flatten()[..., None] @ omega[None]
      return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]
        
  def forward(self, s5):
      B,_,H,W = s5.size()
      s5 = s5.flatten(2).permute(0, 2, 1)
      x = self.projection(s5)
      if self.training:
          pos_embed = self.build_2d_sincos_position_embedding(
              w=W, 
              h=H, 
              embed_dim=self.hidden_dim
          ).to(x.device)
      else:
          pos_embed = getattr(self, 'pos_embed', None).to(x.device)
      x = self.transformer_encoder(x, pos_emb = pos_embed)
      x = x.view(BATCH_SIZE, 20, 20, self.hidden_dim)
      x=x.permute(0, 3, 1, 2)
      return x

class EDF_FAM(nn.Module):
    def __init__(self, in_channels):
        ## first path
        self.reluAct = nn.ReLU()
        self.conv1_1 = nn.Conv2d(2*in_channels, in_channels, kernel_size = 1)
        self.DCN = DCNv2(in_channels, in_channels, act=self.reluAct)
        ## second path
        self.conv2_1 = Conv(2*in_channels, in_channels, k = 1, act=self.reluAct)
        self.glob_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv2_2 = nn.Conv1d(in_channels, in_channels, kernel_size=7)
        self.conv2_3 = nn.Conv1d(in_channels, in_channels, kernel_size=5)
        self.conv2_4 = nn.Conv1d(in_channels, in_channels, kernel_size=3)
        self.conv2_5 = nn.Conv2d(3*in_channels, in_channels, kernel_size=1)
        
        # combined now 
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, y):
        cat = torch.cat([x,y], dim = 1)
        ## first path 
        p1 = self.conv1_1(cat)
        p1 = self.DCN(p1)
        p1 = self.DCN(p1)
        ## second path
        p2 = self.conv2_1(cat)
        p2 = self.glob_avg_pool(p2)
        p2_1 = self.conv2_2(p2)
        p2_2 = self.conv2_3(p2)
        p2_3 = self.conv2_4(p2)
        p2 = torch.cat([p2_1, p2_2, p2_3], dim=1)
        p2 = self.conv2_5(p2)

        
        merge = p1 + p2
        merge = self.sigmoid(merge)
        z1 = merge * x
        z2 = (1 - merge) * y
        out = torch.cat([z1,z2], dim=1)
        return out

class FastResidual1(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1_1 = Conv(in_channels, in_channels//2, k = 1)
        self.conv1_2 = Conv(in_channels//2, in_channels, k = 3, s = 2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_2 = Conv(in_channels, in_channels//2, k = 1)
        self.conv3_2 = Conv(in_channels, in_channels//2, k = 1)
        ## value n_divs is hyperparameter  (arbitrarily chosen)
        self.pConv = Partial_conv3(2 * in_channels,4, 'split_cat')# https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.pdf
        self.conv1_3 = Conv(2*in_channels, 2*in_channels, k=1)
        self.conv1_4 = Conv(2*in_channels, 2*in_channels, k=1)
    def forward(self, x):
        path1 = self.conv1_1(x)
        path1 = self.conv1_2(path1)
        path2 = self.avg_pool(x)
        path2 = self.conv2_2(path2)
        path3 = self.max_pool(x)
        path3 = self.conv3_2(path3)
        mpd1 = torch.cat([path1, path2, path3], dim = 1)
        fasterNet = self.pConv(mpd1)
        fasterNet = self.conv1_3(fasterNet)
        fasterNet = self.conv1_4(fasterNet)
        output = fasterNet + mpd1
        return output
class FastResidual2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1_1 = Conv(in_channels, in_channels//2, k = 1)
        self.conv1_2 = Conv(in_channels//2, in_channels, k = 3, s = 2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_2 = Conv(in_channels, in_channels//2, k = 1)
        self.conv3_2 = Conv(in_channels, in_channels//2, k = 1)
        ## value n_divs is hyperparameter  (arbitrarily chosen)
        self.pConv = Partial_conv3(in_channels,4, 'split_cat')# https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.pdf
        self.conv1_3 = Conv(in_channels, in_channels, k=1)
        self.conv1_4 = Conv(in_channels, in_channels, k=1)
    def forward(self, x):
        path1 = self.conv1_1(x)
        path1 = self.conv1_2(path1)
        path2 = self.avg_pool(x)
        path2 = self.conv2_2(path2)
        path3 = self.max_pool(x)
        path3 = self.conv3_2(path3)
        path23 = torch.cat([path2, path3], dim = 1)
        mpd2 = path1 + path23
        fasterNet = self.pConv(mpd2)
        fasterNet = self.conv1_3(fasterNet)
        fasterNet = self.conv1_4(fasterNet)
        output = fasterNet + mpd2
        return output
class DETR_Backbone(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = Conv(num_channels, 32, k=3)
        self.conv2 = Conv(32, 64, k=3, s=2)
        self.c2f1 = C2f(64, 64, n=3, shortcut = True) # set shortcut to true for further regularization
        self.FR11 = FastResidual1(64)
        self.c2f2 = C2f(128, 128, n=3, shortcut=True) # set shortcut to true for further regularization
        self.FR21 = FastResidual2(128)
        self.FR12 = FastResidual1(128)
        self.FR22 = FastResidual2(256)
        self.final1 = Conv(128, 256, k=1)
        self.final2 = Conv(128, 256, k=1)
        self.final3 = Conv(256, 256, k=1)
        self.final4 = Conv(256, 256, k=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c2f1(x)
        x = self.FR11(x)
        firstOutput = self.c2f2(x) ## goes to neck 

        secondOutput = self.FR21(firstOutput) ## goes to neck
        thirdOutput = self.FR12(secondOutput) ## goes to neck
        fourthOutput = self.FR22(thirdOutput) ## goes to neck
        firstOutput = self.final1(firstOutput)
        secondOutput = self.final2(secondOutput)
        thirdOutput = self.final3(thirdOutput)
        fourthOutput = self.final4(fourthOutput)
        return firstOutput, secondOutput, thirdOutput, fourthOutput
def positional_encoding_2d(height, width, d_model):
    """Creates a 2D sine-cosine positional encoding."""
    assert d_model % 2 == 0, "d_model must be even for sine-cosine encoding"
    
    pe = torch.zeros(height, width, d_model)  # (H, W, C)
 
    y_pos = torch.arange(height, dtype=torch.float32).unsqueeze(1)  # (H, 1)
    x_pos = torch.arange(width, dtype=torch.float32).unsqueeze(0)   # (1, W)
    
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

    pe[:, :, 0::2] = torch.sin(y_pos * div_term).unsqueeze(1)  # Apply to y
    pe[:, :, 1::2] = torch.cos(x_pos * div_term).unsqueeze(0)  # Apply to x
    
    return pe  # Shape: (H, W, C)

class DETR_Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 512
        self.obj_queries = nn.Parameter(torch.randn(100, BATCH_SIZE, self.d_model)) ## 100 object queries (predict 100 bboxes)
        self.queryPosEmbed = nn.Parameter(torch.randn(100, BATCH_SIZE, self.d_model))
        nn.init.xavier_uniform_(self.obj_queries)
        nn.init.xavier_uniform_(self.queryPosEmbed)
        self.decoder = RTDETRDecoder(num_classes=1)
        self.encoder = AIFI()
        self.conv1 = Conv(256, 256)
        
        pass
    def forward(self, outputs):
        first = outputs[0]
        second = outputs[1]
        third = outputs[2]
        fourth = outputs[3]
        print(fourth.shape)
        x = self.encoder(fourth)
        print(x.shape)
        ## ...
        # feature_maps = torch.randn(batch_size, channels /// should match decoder d_model, height, width)
        # memory = feature_maps.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        # x = self.obj_queries
        # pos_encoding = positional_encoding_2d(feature_maps.size(2), feature_maps.size(3), self.d_model)  # (H, W, C)
        # pos_embed = pos_encoding.flatten(0, 1).unsqueeze(1)  
        # pos_embed = pos_embed.expand(-1, BATCH_SIZE, -1)
        # out = self.decoder(x, memory, pos_embed = pos_embed, query_pos_embed = self.queryPosEmbed)
        # return out
        #  
        #  Now feed those into decoder as (x, memory, pos_embed = feature maps pos enc, query_pos_embed = obj query pos enc)
        ## todo: implement neck
model = DETR_Backbone(4)
neck = DETR_Neck()
image = cv2.imread("rgb.jpg")  # Replace with your image path
transform = transforms.Compose([
    transforms.ToTensor()  # Converts to tensor and normalizes to [0,1]
])

# Convert BGR to RGB (PyTorch models expect RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
image = image.astype("float32") / 255.0

# Convert to PyTorch tensor and rearrange dimensions (H, W, C) → (C, H, W)
tensor_image = torch.from_numpy(image).permute(2, 0, 1)  # Convert to (C, H, W)
tensor_image = tensor_image.unsqueeze(0)
print(tensor_image.shape)
new_channel = torch.randn(1,1,640,640)
tensor_image = torch.cat([tensor_image,new_channel], dim=1)
x = (time.time())
out = model(tensor_image)
out = neck(out)
print(time.time() - x)
print(out.shape)