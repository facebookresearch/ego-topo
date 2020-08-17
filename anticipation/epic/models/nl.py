import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import dgl

class LFB_NL(nn.Module):
    def __init__(self, clip_in_dim, lfb_in_dim, h_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.clip_process = nn.Sequential(
                                nn.Conv3d(clip_in_dim, h_dim, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
                                nn.Dropout(0.2)) 
        self.lfb_process = nn.Sequential(
                                nn.Conv3d(lfb_in_dim, h_dim, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
                                nn.Dropout(0.2)) 



        self.theta = nn.Conv3d(h_dim, h_dim, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        self.phi = nn.Conv3d(h_dim, h_dim, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        self.g = nn.Conv3d(h_dim, h_dim, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))

        self.out = nn.Conv3d(h_dim, h_dim, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))

        self.drop = nn.Dropout(0.2)
        self.h_dim = h_dim


    # (B, 2048) --> (B, 512, 1, 1, 1)
    def process_clip(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)
        x = self.clip_process(x)
        return x

    # (B, N, 2048) --> (B, 512, N, 1, 1)
    def process_lfb(self, x):
        x = x.transpose(1, 2)
        x = x.view(x.shape[0], x.shape[1], x.shape[2], 1, 1)
        x = self.lfb_process(x)
        return x

    # def nl_core(self, A, B):
    #     num_lfb_feat = B.shape[2]
    #     theta = self.theta(A)
    #     phi = self.phi(B)
    #     g = self.g(B)

    #     theta_shape = theta.shape
    #     theta, phi, g = theta.view(-1, self.h_dim, 1), phi.view(-1, self.h_dim, num_lfb_feat), g.view(-1, self.h_dim, num_lfb_feat)

    #     theta_phi = torch.bmm(theta.transpose(1, 2), phi) # (B, 512, 1) * (B, 512, num_lfb_feats) => (B, 1, num_lfb_feats)
    #     theta_phi_sc = theta_phi * (self.h_dim**-.5)
    #     p = F.softmax(theta_phi_sc, dim=-1)

    #     t = torch.bmm(g, p.transpose(1, 2))
    #     t = t.view(theta_shape)

    #     out = F.layer_norm(t, t.shape[1:])
    #     out = F.relu(out)
    #     out = self.out(out)
    #     out = self.drop(out)

    #     return out

    # # (B, 2048), (B, N, 2048)
    # def forward(self, clip, lfb):
    #     A, B = self.process_clip(clip), self.process_lfb(lfb) # (B, 512, 1), (B, 512, N)
    #     for _ in range(self.num_layers):
    #         out = self.nl_core(A, B)
    #         # out = out + A
    #         A = out
    #     out = out.view(out.shape[0], -1)
    #     return out


    def nl_core(self, A, B, bg):
        num_lfb_feat = B.shape[2]
        theta = self.theta(A)
        phi = self.phi(B)
        g = self.g(B)

        theta_shape = theta.shape
        theta, phi, g = theta.view(-1, self.h_dim, 1), phi.view(-1, self.h_dim, num_lfb_feat), g.view(-1, self.h_dim, num_lfb_feat)

        theta_phi = torch.bmm(theta.transpose(1, 2), phi) # (B, 512, 1) * (B, 512, num_lfb_feats) => (B, 1, num_lfb_feats)
        theta_phi_sc = theta_phi * (self.h_dim**-.5)
        p = F.softmax(theta_phi_sc, dim=-1) # (B, 1, N)

        mask = torch.ByteTensor([graph.ndata['next_status'].sum()>0 for graph in dgl.unbatch(bg)]).to(p.device)
        p_target = torch.LongTensor([graph.ndata['next_status'].argmax().item() for graph in dgl.unbatch(bg)]).to(p.device)
        loss = F.cross_entropy(theta_phi_sc[mask].squeeze(1), p_target[mask])

        t = torch.bmm(g, p.transpose(1, 2))
        t = t.view(theta_shape)

        out = F.layer_norm(t, t.shape[1:])
        out = F.relu(out)
        out = self.out(out)
        out = self.drop(out)

        return out, 0

    # (B, 2048), (B, N, 2048)
    def forward(self, clip, lfb, bg):
        A, B = self.process_clip(clip), self.process_lfb(lfb) # (B, 512, 1), (B, 512, N)
        total_loss = 0
        for _ in range(self.num_layers):
            out, loss = self.nl_core(A, B, bg)
            out = out + A
            total_loss = total_loss + loss
            A = out

        total_loss = total_loss/self.num_layers

        out = out.view(out.shape[0], -1)
        return out, loss
