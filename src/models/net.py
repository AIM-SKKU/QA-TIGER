import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Dict

from src.models.encoders import CLIP_TEncoder
from src.models.modules import (
    Projection, QstGrounding,
    TempMoE, AVQCrossAttn,
    PatchSelecter
)


class QA_TIGER(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 video_dim: int = 512,
                 patch_dim: int = 768,
                 audio_dim: int = 128,
                 topK: int = 3,
                 num_experts: int = 10,
                 late_fusion: bool = False,
                 nce_loss: bool = False,
                 encoder_type: str = 'ViT-L/14@336px',
                 **kwargs
    ):
        super(QA_TIGER, self).__init__()
        
        self.nce_loss = nce_loss
        self.late_fusion =late_fusion
        self.audio_proj = Projection(audio_dim, d_model)
        self.video_proj = Projection(video_dim, d_model)
        self.patch_proj = Projection(patch_dim, d_model)
        self.words_proj = Projection(video_dim, d_model)
        self.quest_proj = Projection(video_dim, d_model)

        self.quest_encoder = CLIP_TEncoder(encoder_type)
        self.quest_encoder.freeze()

        self.crs_attn = AVQCrossAttn(d_model, 8)
        self.patch_selecter = PatchSelecter(d_model, 8)
        self.quest_grounding = QstGrounding(d_model, 8)
        self.at_aggregator = TempMoE(d_model, 8, topK=topK, n_experts=num_experts)
        self.vt_aggregator = TempMoE(d_model, 8, topK=topK, n_experts=num_experts, vis_branch=True)
        self.head_act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(d_model, 42)
        
        ## freeze & init
        self.audio_proj.apply(self.init_weight)
        self.video_proj.apply(self.init_weight)
        self.words_proj.apply(self.init_weight)
        self.quest_proj.apply(self.init_weight)
        self.patch_proj.apply(self.init_weight)
        self.head.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def sub_forward(self, 
                    reshaped_data: Dict[str, Tensor],
                    prefix: str = ''
    ):
        
        quest = reshaped_data[f'{prefix}quest']
        audio = reshaped_data[f'{prefix}audio']
        video = reshaped_data[f'{prefix}video']
        patch = reshaped_data[f'{prefix}patch'] if f'{prefix}patch' in reshaped_data else None
        
        if prefix == 'n_':
            words = None
            quest = None
        else:
            if quest.dtype in (torch.float32, torch.float64):
                quest = quest.squeeze(1)
                words = None
            else:
                quest, words = self.quest_encoder(quest)
                quest = quest.squeeze(1)
        
        return quest, words, audio, video, patch
    
    
    def forward(self, reshaped_data: Dict[str, Tensor]):
        '''
            input audio shape:      [B, T, AC]
            input pos_frames shape: [B, T, VC, FH, FW]
            input question shape:   [B, D]
            input neg_frames shape: [B, T, VC, FH, FW]
        '''
        return_dict = {}

        quest, words, audio, video, patch =self.sub_forward(reshaped_data, prefix='')
            
        # Projection
        audio = self.audio_proj(audio) # [B, T, D]
        video = self.video_proj(video) # [B, T, D]
        words = self.words_proj(words) # [B, 77, D]
        quest = self.quest_proj(quest) # [B, D]
        patch = self.patch_proj(patch) # [B, T, P, D]
        
        audio, video = self.crs_attn(audio, video, words) # [B, T, D], [B, T, D]
        patch = self.patch_selecter(patch, audio, video)  # [B, T, D]
        a_global = self.at_aggregator(quest, audio)
        ap_global, vp_global = self.vt_aggregator(quest, video, patch)
        fusion = self.quest_grounding(quest, [ap_global, vp_global])
        fusion = self.quest_grounding(quest, [fusion.unsqueeze(1), a_global])

        fusion = self.head_act(fusion)
        output = self.head(fusion)
        return_dict.update({'out': output})
        return return_dict
