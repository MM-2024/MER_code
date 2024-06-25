'''
Description: unimodal encoder + concat + attention fusion
'''
import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder
from .cross_model.bert_components import BertAttention, LinearLayer, BertSelfAttention, TrainablePositionalEncoding
from . import config
class Attention_TOPN(nn.Module):
    def __init__(self, args):
        super(Attention_TOPN, self).__init__()
        
        feat_dims = args.audio_dim # store topn feat dims
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout = args.dropout
        hidden_dim = args.hidden_dim
        self.grad_clip = args.grad_clip
        self.feat_dims = feat_dims

        ## --------------------------------------------------- ##
        # self.encoders = []
        # for dim in feat_dims:
        #     self.encoders.append(MLPEncoder(dim, hidden_dim, dropout).cuda()) # list 不能 cuda，还是这么操作吧
         ## --------------------------------------------------- ##
        # debug: list 不能传 cuda 和不能传梯度，是不是一个意思呢？ => yes
        assert len(feat_dims) <= 3 * 6
        if len(feat_dims) >= 1:  self.encoder0  = MLPEncoder(feat_dims[0],  hidden_dim, dropout)
        if len(feat_dims) >= 2:  self.encoder1  = MLPEncoder(feat_dims[1],  hidden_dim, dropout)
        if len(feat_dims) >= 3:  self.encoder2  = MLPEncoder(feat_dims[2],  hidden_dim, dropout)
        if len(feat_dims) >= 4:  self.encoder3  = MLPEncoder(feat_dims[3],  hidden_dim, dropout)
        if len(feat_dims) >= 5:  self.encoder4  = MLPEncoder(feat_dims[4],  hidden_dim, dropout)
        if len(feat_dims) >= 6:  self.encoder5  = MLPEncoder(feat_dims[5],  hidden_dim, dropout)
        if len(feat_dims) >= 7:  self.encoder6  = MLPEncoder(feat_dims[6],  hidden_dim, dropout)
        if len(feat_dims) >= 8:  self.encoder7  = MLPEncoder(feat_dims[7],  hidden_dim, dropout)
        if len(feat_dims) >= 9:  self.encoder8  = MLPEncoder(feat_dims[8],  hidden_dim, dropout)
        if len(feat_dims) >= 10: self.encoder9  = MLPEncoder(feat_dims[9],  hidden_dim, dropout)
        if len(feat_dims) >= 11: self.encoder10 = MLPEncoder(feat_dims[10], hidden_dim, dropout)
        if len(feat_dims) >= 12: self.encoder11 = MLPEncoder(feat_dims[11], hidden_dim, dropout)
        if len(feat_dims) >= 13: self.encoder12 = MLPEncoder(feat_dims[12], hidden_dim, dropout)
        if len(feat_dims) >= 14: self.encoder13 = MLPEncoder(feat_dims[13], hidden_dim, dropout)
        if len(feat_dims) >= 15: self.encoder14 = MLPEncoder(feat_dims[14], hidden_dim, dropout)
        if len(feat_dims) >= 16: self.encoder15 = MLPEncoder(feat_dims[15], hidden_dim, dropout)
        if len(feat_dims) >= 17: self.encoder16 = MLPEncoder(feat_dims[16], hidden_dim, dropout)
        if len(feat_dims) >= 18: self.encoder17 = MLPEncoder(feat_dims[17], hidden_dim, dropout)
        
        self.attention_mlp = MLPEncoder(hidden_dim * len(feat_dims), hidden_dim, dropout)
        self.fc_att   = nn.Linear(hidden_dim, len(feat_dims))
        
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)
        
        "新加注意力"
        self.video_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_video_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)
        "config.max_video_l 是最大视频帧数 "
        self.au_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_au_l,
                                                         hidden_size=config.hidden_size, dropout=config.input_drop)
        "config.max_au_l 是音频最大长度"
        self.video_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.audio_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.video_encoder = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))
        self.audio_encoder = copy.deepcopy(self.video_encoder)
        cross_att_cfg = edict(hidden_size=config.hidden_size, num_attention_heads=config.n_heads,
                              attention_probs_dropout_prob=config.drop)
        self.video_cross_att = BertSelfAttention(cross_att_cfg)
        self.video_cross_layernorm = nn.LayerNorm(config.hidden_size)
        
        self.audio_cross_att = BertSelfAttention(cross_att_cfg)
        self.audio_cross_layernorm = nn.LayerNorm(config.hidden_size)
        
    
    def forward(self, batch):
        hiddens = []
        ## --------------------------------------------------- ##
        # for ii, encoder in enumerate(self.encoders):
        #     hiddens.append(encoder(batch[f'feat{ii}']))
        ## --------------------------------------------------- ##
        if len(self.feat_dims) >= 1: hiddens.append(self.encoder0(batch[f'feat0']))
        if len(self.feat_dims) >= 2: hiddens.append(self.encoder1(batch[f'feat1']))
        if len(self.feat_dims) >= 3: hiddens.append(self.encoder2(batch[f'feat2']))
        if len(self.feat_dims) >= 4: hiddens.append(self.encoder3(batch[f'feat3']))
        if len(self.feat_dims) >= 5: hiddens.append(self.encoder4(batch[f'feat4']))
        if len(self.feat_dims) >= 6: hiddens.append(self.encoder5(batch[f'feat5']))
        if len(self.feat_dims) >= 7: hiddens.append(self.encoder6(batch[f'feat6']))
        if len(self.feat_dims) >= 8: hiddens.append(self.encoder7(batch[f'feat7']))
        if len(self.feat_dims) >= 9: hiddens.append(self.encoder8(batch[f'feat8']))
        if len(self.feat_dims) >= 10: hiddens.append(self.encoder9(batch[f'feat9']))
        if len(self.feat_dims) >= 11: hiddens.append(self.encoder10(batch[f'feat10']))
        if len(self.feat_dims) >= 12: hiddens.append(self.encoder11(batch[f'feat11']))
        if len(self.feat_dims) >= 13: hiddens.append(self.encoder12(batch[f'feat12']))
        if len(self.feat_dims) >= 14: hiddens.append(self.encoder13(batch[f'feat13']))
        if len(self.feat_dims) >= 15: hiddens.append(self.encoder14(batch[f'feat14']))
        if len(self.feat_dims) >= 16: hiddens.append(self.encoder15(batch[f'feat15']))
        if len(self.feat_dims) >= 17: hiddens.append(self.encoder16(batch[f'feat16']))
        if len(self.feat_dims) >= 18: hiddens.append(self.encoder17(batch[f'feat17']))

        multi_hidden1 = torch.cat(hiddens, dim=1) # [32, 384]
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 2) # [32, 3, 1]

        multi_hidden2 = torch.stack(hiddens, dim=2) # [32, 128, 3]
        fused_feat = torch.matmul(multi_hidden2, attention)  # [32, 128, 3] * [32, 3, 1] = [32, 128, 1]

        features  = fused_feat.squeeze(axis=2) # [32, 128] => 解决batch=1报错的问题
        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.tensor(0).cuda()

        return features, emos_out, vals_out, interloss
    
    def encode_context(self, video_feat, video_mask, sub_feat, sub_mask, return_mid_output=False):
        # encoding video and audio features, respectively
        encoded_video_feat = self.encode_input(video_feat, video_mask, self.video_input_proj, self.video_encoder,
                                               self.video_pos_embed)
        encoded_audio_feat = self.encode_input(audio_feat, audio_mask, self.audio_input_proj, self.audio_encoder,
                                             self.au_pos_embed)
        # cross encoding audio features
        x_encoded_video_feat = self.cross_context_encoder(encoded_video_feat, video_mask, encoded_sub_feat, sub_mask,
                                                          self.video_cross_att, self.video_cross_layernorm)  # (N, L, D)
        # cross encoding video features
        x_encoded_audio_feat = self.cross_context_encoder(encoded_sub_feat, sub_mask, encoded_video_feat, video_mask,
                                                        self.audio_cross_att, self.audio_cross_layernorm)  # (N, L, D)
       
        return x_encoded_video_feat, x_encoded_sub_feat
    
    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        return encoder_layer(feat, mask)  # (N, L, D_hidden)
    
    @staticmethod
    def cross_context_encoder(main_context_feat, main_context_mask, side_context_feat, side_context_mask,
                              cross_att_layer, norm_layer):
        """
        Args:
            main_context_feat: (N, Lq, D)
            main_context_mask: (N, Lq)
            side_context_feat: (N, Lk, D)
            side_context_mask: (N, Lk)
            cross_att_layer: cross attention layer
            norm_layer: layer norm layer
        """
        cross_mask = torch.einsum("bm,bn->bmn", main_context_mask, side_context_mask)  # (N, Lq, Lk)
        cross_out = cross_att_layer(main_context_feat, side_context_feat, side_context_feat, cross_mask)  # (N, Lq, D)
        residual_out = norm_layer(cross_out + main_context_feat)
        return residual_out