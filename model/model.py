import copy
import torch
from torch import nn
from torch.nn import functional as F
from .attention import MultiHeadedAttention
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .shared import Embeddings, PositionwiseFeedForward


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        #self.min = torch.zeros((64))
        #self.min[53:56] = worldMin
        #self.min[56:59] = scalingMin
        #self.min = torch.unsqueeze(torch.unsqueeze(self.min.repeat(2**stacking),0),0).cuda()
        #self.max = torch.ones((64))
        #self.max[53:56] = worldMax
        #self.max[56:59] = scalingMax
        #self.max = torch.unsqueeze(torch.unsqueeze(self.max.repeat(2**stacking),0),0).cuda()
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        #src = (src - self.min) / (self.max - self.min)
        #tgt = (tgt - self.min) / (self.max - self.min)
        tgt =  self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
        return tgt
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, g_len):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, g_len)

    def forward(self, x):
        return self.proj(x)
    

def make_model(stacking, src_g_len=64, tgt_g_len=64, N=2, 
               d_model=32, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_model*2, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
                                            c(ff),
                                            c(ff),
        Generator(d_model, tgt_g_len))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model