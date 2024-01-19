import time
import torch
import numpy as np
from torch.autograd import Variable
from model.shared import subsequent_mask
from model.model import make_model, EncoderDecoder

START_GAUSSIAN = torch.zeros(64,dtype=torch.float32)
START_GAUSSIAN[60] = 1
END_GAUSSIAN = torch.zeros(64,dtype=torch.float32)
END_GAUSSIAN[63] = 1

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=-99):
        self.src = src
        self.src_mask = (src.unsqueeze(-3)[:, :, :, 0] != pad)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-3)[:, :, :, 0]
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt_mask.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, opt=None):
        self.generator = generator
        self.criterion = torch.nn.L1Loss()
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)

        loss = self.criterion(x, y) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm
    
def closestFuture(pred, tgt):
    near = torch.abs(tgt - pred)
    close = torch.sum(near, dim=-1)
    closest, closest_idx = torch.min(close, dim=-1)
    return closest, closest_idx

def closestFutures(pred, tgt):
    rPred, rTgt = pred, tgt
    loss = 0
    for _ in range(pred.shape[1]):
        mask = torch.ones(rPred.shape[:-1], dtype=torch.bool)
        closest, idx = closestFuture(rPred[:, 0:1], rTgt)
        rPred = rPred[:, 1:]
        for i in range(pred.shape[0]):
            mask[i,idx[i]] = False
        rTgt = rTgt[mask].reshape(pred.shape[0], rPred.shape[1], rPred.shape[2])
        loss += closest
    return loss

        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def data_gen(batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.rand(batch, 10, 64).astype(np.float32))
        data[:, :, 60:] = 0
        data[:, 0, 60] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt)

def run_epoch(data_iter, model : EncoderDecoder, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        batch : Batch = batch
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            #print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    #(i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = start_symbol.unsqueeze(0).unsqueeze(0)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        next_word = prob.data[0]
        ys = torch.cat([ys, 
                        next_word.unsqueeze(0).unsqueeze(0)], dim=1)
    return ys



V = 64
model = make_model(V, V, N=2)


model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

model.eval()
src = Variable(torch.torch.from_numpy(np.random.rand(1, 3, 64).astype(np.float32)))
src_mask = Variable(torch.ones(1, 1, 3) )
print(src)
pred = greedy_decode(model, src, src_mask, max_len=3, start_symbol=START_GAUSSIAN)
print(pred)
print(torch.sum(torch.abs(src - pred)))

for epoch in range(10):
    model.train()
    run_epoch(data_gen(30, 20), model, 
              SimpleLossCompute(model.generator, model_opt))
    model.eval()
    #print(run_epoch(data_gen(30, 5), model, 
                    #SimpleLossCompute(model.generator, None)))
    
model.eval()
pred = greedy_decode(model, src, src_mask, max_len=3, start_symbol=START_GAUSSIAN)
print(pred)
print(torch.sum(torch.abs(src - pred)))

