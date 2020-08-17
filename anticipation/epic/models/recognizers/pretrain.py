from mmaction.models import builder
from mmaction.models.recognizers import BaseRecognizer
from mmaction.models.registry import RECOGNIZERS

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence, PackedSequence

import numpy as np

@RECOGNIZERS.register_module
class Pretrain(BaseRecognizer):
    def __init__(self, train_cfg=None, test_cfg=None):
        super().__init__()

        self.rnn = nn.LSTM(2048, 2048, num_layers=2, batch_first=True)

        self.mlp = nn.Sequential(
                    nn.Linear(2048, 2048),
                    nn.ReLU(True),
                    nn.Linear(2048, 2048),
                    nn.ReLU(True),
                    nn.Linear(2048, 2048),
                    nn.ReLU(True),
                    )

        self.vdist = nn.Linear(2048, 125)
        self.ndist = nn.Linear(2048, 352)
        self.idist = nn.Linear(2048, 250)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights()


    def init_weights(self):
        super().init_weights()
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
        for module in [self.vdist, self.ndist, self.idist]:
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)

    # def dist_loss(self, inp, target, dist_net, src):

    #     # src: 0 if it's from a visit, 1 if it's from an annotated clip

    #     mask = target.sum(1)>0
    #     inp, target = inp[mask], target[mask]

    #     pred = dist_net(inp)
    #     pred = torch.sigmoid(pred)

    #     target_bin = (target>0).float()
    #     loss = F.binary_cross_entropy(pred, target_bin, reduction='none')

    #     target_max = target.argmax(1)
    #     mask = []
    #     for b in range(inp.shape[0]):

    #         # from visit data -- accumulate loss from all classes
    #         if src[b].item()==0:
    #             mask.append(torch.ones(target.shape[1]))

    #         # from clip dataset -- accumulate loss from the single class we have a label for
    #         elif src[b].item()==1:
    #             m = torch.zeros(target.shape[1])
    #             m[target_max[b].item()] = 1
    #             mask.append(m)

    #     mask = torch.stack(mask, 0).to(inp.device) # (B, num_classes)
    #     loss = loss*mask
    #     loss = loss.sum(1)/mask.sum(1)

    #     return loss


    def dist_loss(self, inp, target, dist_net, src):

        mask = target.sum(1)>0
        inp, target = inp[mask], target[mask]

        pred = dist_net(inp)

        _target = []
        for b in range(inp.shape[0]):
            candidates = target[b].nonzero().squeeze(1).cpu().numpy().tolist()
            _target.append(candidates[np.random.randint(len(candidates))])
        _target = torch.LongTensor(_target).to(pred.device)

        loss = F.cross_entropy(pred, _target)
       
        return loss

    def forward_train(self, num_modalities, img_meta, **kwargs): 

        losses = {}

        feats, length = kwargs['feature'], kwargs['length'].squeeze(1)

        # # MLP
        # feats = feats.sum(1)/length.float().unsqueeze(1)
        # out_feats = self.mlp(feats)

        K = np.random.randint(1, 4)

        feat_in = feats[:, :-K] # (B, T-1)
        verbs = kwargs['verb_labels'][:, K:] # (B, T-1)
        nouns = kwargs['noun_labels'][:, K:] # (B, T-1)

        # LSTM
        packed_input = pack_padded_sequence(feat_in, length-K, batch_first=True, enforce_sorted=False)
        out, (ht, ct) = self.rnn(packed_input)

        packed_verbs = pack_padded_sequence(verbs, length-K, batch_first=True, enforce_sorted=False)
        packed_nouns = pack_padded_sequence(nouns, length-K, batch_first=True, enforce_sorted=False)

        verb_pred = self.vdist(out.data)
        noun_pred = self.ndist(out.data)

        losses['verb_loss'] = F.cross_entropy(verb_pred, packed_verbs.data)
        losses['noun_loss'] = F.cross_entropy(noun_pred, packed_nouns.data)


        # losses['vdist_loss'] = self.dist_loss(out_feats, kwargs['verb_dist'], self.vdist, kwargs['src'])
        # losses['ndist_loss'] = self.dist_loss(out_feats, kwargs['noun_dist'], self.ndist, kwargs['src'])
        # losses['idist_loss'] = self.dist_loss(out_feats, kwargs['int_dist'], self.idist, kwargs['src'])

        return losses

    def forward_test(self, num_modalities, img_meta, **kwargs):
        x = kwargs['feature']
        ret = x.cpu().numpy()
        return ret
