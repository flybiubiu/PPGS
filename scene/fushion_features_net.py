import torch
import torch.nn as nn
from net_modules.embedder import *
from net_modules.basic_mlp import lin_module


class FushionModel(nn.Module):
    def __init__(self,
                fin_dim,
                out_dim,
                view_dim = None,
                multires = [10, 0]
                ):
        super().__init__()
        self.embed_fns=[]
        #print(self.pre_compc, self.use_pencoding)True [True, False]
        #print(self.use_decode_with_pos)False
        #print(multires)[10, 0]

        embed_fn, input_ch = get_embedder(multires[0])
        #print(embed_fn)<function get_embedder.<locals>.embed at 0x7ee7954f28b0>
        #print(input_ch)63
        pin_dim = input_ch
        self.embed_fns.append(embed_fn)

        en_dims_v = [64, 96, 128]
        self.features = lin_module(out_dim , 128, en_dims_v, multires[0],act_fun=nn.ReLU())
        en_dims = [128, 96, 64]
        #print(fin_dim, pin_dim, pfin_dim, en_dims, de_dims)48 63 48 [128, 96, 64] [48, 48]
        self.encoder=lin_module(fin_dim,out_dim,en_dims,multires[0],act_fun=nn.ReLU())



                
    def forward(self, f, xyzs,view_direction=None):
        f = self.features(f)
        xyzs_feature = self.embed_fns[0](xyzs)
        inx=torch.cat([f,xyzs_feature],dim=1)# 191, 8
        oute= self.encoder(inx)
        #print(oute.shape)torch.Size([100000, 48])
        return oute

'''
    def forward_cache(self, inp,view_direction=None):
        oinp=inp
        if  self.use_pencoding[0]:
            if self.use_decode_with_pos:
                oinp=inp.clone()
            inp = self.embed_fns[0](inp)
        if  self.use_pencoding[1]:
            view_direction = self.embed_fns[1](view_direction)
        p_num=inp.shape[0]
        if self.pre_compc:
            if self.use_decode_with_pos:
                outc=self.color_decoder(torch.cat([self.cache_outd,oinp],dim=1))
            else:
                outc=self.color_decoder(torch.cat([self.cache_outd,view_direction],dim=1)) #view_direction
            return outc
        return self.cache_outd.reshape([p_num,-1,3])
'''
        

    

        