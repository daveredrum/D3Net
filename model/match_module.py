import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer.attention import MultiHeadAttention
from model.transformer.utils import PositionWiseFeedForward


class MatchModule(nn.Module):
    def __init__(self, cfg,
        lang_size=256, hidden_size=128):
        
        super().__init__() 

        self.num_proposals = cfg.model.max_num_proposal
        self.lang_size = lang_size
        self.hidden_size = hidden_size
        self.det_channel = cfg.model.m

        self.chunk_size = cfg.data.num_des_per_scene

        self.fuse = nn.Sequential(
            nn.Conv1d(lang_size + self.det_channel, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
        )
        # self.match = nn.Conv1d(hidden_size, 1, 1)
        self.match = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, 1)
        )

    def forward(self, data_dict, use_rl=False):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """

        if use_rl:
            # unpack outputs from detection branch
            features = data_dict["proposal_feats_batched"] # batch_size, num_proposal, 128
            objectness_masks = data_dict["proposal_batch_mask"].float().unsqueeze(2) # batch_size, num_proposals, 1

            # unpack sampled outputs from language branch
            # NOTE the batch size is actually the old batch_size x sample_topn
            sampled_feat = data_dict["lang_emb"]["sampled"] # batch_size, lang_size
            
            # unpack baseline outputs from language branch
            baseline_feat = data_dict["lang_emb"]["baseline"] # batch_size, lang_size

            # expand object contexts to match the language features
            sampled_topn = data_dict["sampled_topn"] # e.g. 3
            old_batch_size = features.shape[0]
            old_ids = torch.arange(old_batch_size)
            exp_ids = old_ids.repeat_interleave(sampled_topn)
            features_exp = features[exp_ids]
            objectness_masks_exp = objectness_masks[exp_ids]
            assert features_exp.shape[0] == sampled_feat.shape[0]

            sampled_feat = sampled_feat.unsqueeze(1).repeat(1, self.num_proposals, 1) # batch_size, num_proposals, lang_size
            baseline_feat = baseline_feat.unsqueeze(1).repeat(1, self.num_proposals, 1) # batch_size, num_proposals, lang_size

            # ------ for sampled -------
            # fuse
            features = torch.cat([features_exp, sampled_feat], dim=-1) # batch_size, num_proposals, 128 + lang_size
            features = features.permute(0, 2, 1).contiguous() # batch_size, 128 + lang_size, num_proposals

            # fuse features
            features = self.fuse(features) # batch_size, hidden_size, num_proposals

            # match
            confidences = self.match(features).squeeze(1) # batch_size, num_proposals
            # # NOTE make sure again the empty boxes won"t be part of the discrimination
            # confidences.masked_fill_(objectness_masks_exp.squeeze(-1) == 0, float("-1e30"))

            sampled_ref = confidences

            # ------ for baseline -------
            with torch.no_grad():
                # fuse
                features = torch.cat([features_exp, baseline_feat], dim=-1) # batch_size, num_proposals, 128 + lang_size
                features = features.permute(0, 2, 1).contiguous() # batch_size, 128 + lang_size, num_proposals

                # fuse features
                features = self.fuse(features) # batch_size, hidden_size, num_proposals

                # match
                confidences = self.match(features).squeeze(1) # batch_size, num_proposals
                # # NOTE make sure again the empty boxes won"t be part of the discrimination
                # confidences.masked_fill_(objectness_masks_exp.squeeze(-1) == 0, float("-1e30"))

                baseline_ref = confidences

            data_dict["cluster_ref"] = {
                "sampled": sampled_ref,
                "baseline": baseline_ref
            }

        else:
            # unpack outputs from detection branch
            features = data_dict["proposal_feats_batched"] # batch_size, num_proposal, det_channel
            objectness_masks = data_dict["proposal_batch_mask"].float().unsqueeze(2) # batch_size, num_proposals, 1

            # expand features
            features = features.unsqueeze(1).repeat(1, self.chunk_size, 1, 1) # batch_size, chunk_size, num_proposal, det_channel
            features = features.reshape(-1, self.num_proposals, self.det_channel) # batch_size * chunk_size, num_proposal, det_channel
            objectness_masks = objectness_masks.unsqueeze(1).repeat(1, self.chunk_size, 1, 1) # batch_size, chunk_size, num_proposals, 1
            objectness_masks = objectness_masks.reshape(-1, self.num_proposals, 1) # batch_size * chunk_size, num_proposal, 1

            # unpack outputs from language branch
            lang_feat = data_dict["lang_emb"] # batch_size * chunk_size, lang_size
            lang_feat = lang_feat.unsqueeze(1).repeat(1, self.num_proposals, 1) # batch_size * chunk_size, num_proposals, lang_size

            # fuse
            features = torch.cat([features, lang_feat], dim=-1) # batch_size * chunk_size, num_proposals, det_channel + lang_size
            features = features.permute(0, 2, 1).contiguous() # batch_size * chunk_size, det_channel + lang_size, num_proposals

            # fuse features
            features *= objectness_masks.permute(0, 2, 1).contiguous() # batch_size * chunk_size, det_channel + lang_size, num_proposals
            features = self.fuse(features) # batch_size * chunk_size, hidden_size, num_proposals

            # match
            confidences = self.match(features).squeeze(1) # batch_size * chunk_size, num_proposals
            
            # # NOTE make sure again the empty boxes won"t be part of the discrimination
            # confidences.masked_fill_(objectness_masks.squeeze(-1) == 0, float("-1e30"))

            data_dict["cluster_ref"] = confidences # batch_size * chunk_size, num_proposals

        return data_dict

class TransformerMatchModule(nn.Module):
    def __init__(self, cfg, 
        lang_size=256, hidden_size=128, head=4, depth=2,
        use_dist_weight_matrix=True):
        super().__init__()

        self.use_dist_weight_matrix = use_dist_weight_matrix

        self.num_proposals = cfg.model.max_num_proposal
        self.lang_size = lang_size
        self.hidden_size = hidden_size
        self.head = head
        self.depth = depth - 1
        self.det_channel = cfg.model.m

        self.chunk_size = cfg.data.num_des_per_scene

        self.features_concat = nn.Sequential(
            nn.Conv1d(self.det_channel, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
        )
        self.match = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Conv1d(hidden_size, 1, 1)
        )

        self.lang_fc = nn.Sequential(
            nn.Linear(lang_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.LayerNorm(hidden_size)
        )
        self.lang_self_attn = MultiHeadAttention(d_model=hidden_size, d_k=16, d_v=16, h=head)

        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head) for i in range(depth))
        self.cross_attn = nn.ModuleList(
            MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head) for i in range(depth))  # k, q, v

    def multiplex_attention(self, v_features, l_features, l_masks, dist_weights, attention_matrix_way):
        batch_size, num_words, _ = l_features.shape

        lang_self_masks = l_masks.reshape(batch_size, 1, 1, -1).contiguous().repeat(1, self.head, num_words, 1)

        l_features = self.lang_fc(l_features)
        l_features = self.lang_self_attn(l_features, l_features, l_features, lang_self_masks)

        lang_cross_masks = l_masks.reshape(batch_size, 1, 1, -1).contiguous().repeat(1, self.head, self.num_proposals, 1)
        v_features = self.cross_attn[0](v_features, l_features, l_features, lang_cross_masks)

        for _ in range(self.depth):
            v_features = self.self_attn[_+1](v_features, v_features, v_features, attention_weights=dist_weights, way=attention_matrix_way)
            v_features = self.cross_attn[_+1](v_features, l_features, l_features, lang_cross_masks)

        # print("feature1", feature1.shape)
        # match
        v_features_agg = v_features.permute(0, 2, 1).contiguous()

        confidence = self.match(v_features_agg).squeeze(1)  # batch_size, num_proposals

        return confidence

    def forward(self, data_dict, use_rl=False):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.use_dist_weight_matrix:
            # Attention Weight
            # objects_center = data_dict["bbox_center"]
            objects_center = data_dict["proposal_center_batched"]
            N_K = objects_center.shape[1]
            center_A = objects_center[:, None, :, :].repeat(1, N_K, 1, 1)
            center_B = objects_center[:, :, None, :].repeat(1, 1, N_K, 1)
            dist = (center_A - center_B).pow(2)
            # print(dist.shape, "<< dist shape", flush=True)
            dist = torch.sqrt(torch.sum(dist, dim=-1))[:, None, :, :]
            dist_weights = 1 / (dist+1e-2)
            norm = torch.sum(dist_weights, dim=2, keepdim=True)
            dist_weights = dist_weights / norm

            # zeros = torch.zeros_like(dist_weights)
            dist_weights = torch.cat([dist_weights for _ in range(self.head)], dim=1).detach()
            # dist_weights = torch.cat([dist_weights, -dist, zeros, zeros], dim=1).detach()
            # dist_weights = torch.cat([-dist, -dist, zeros, zeros], dim=1).detach()
            attention_matrix_way = "add"
        else:
            dist_weights = None
            attention_matrix_way = "mul"

        # dist_weights = None

        # object size embedding
        # print(data_dict.keys())
        # features = data_dict["bbox_feature"]
        features = data_dict["proposal_feats_batched"]
        # features = features.permute(1, 2, 0, 3)
        # B, N = features.shape[:2]
        # features = features.reshape(B, N, -1).permute(0, 2, 1)
        features = features.permute(0, 2, 1)
        features = self.features_concat(features).permute(0, 2, 1)
        batch_size, num_proposal = features.shape[:2]

        # objectness_masks = data_dict["bbox_mask"].float().unsqueeze(2) # batch_size, num_proposals, 1
        objectness_masks = data_dict["proposal_batch_mask"].float().unsqueeze(2) # batch_size, num_proposals, 1

        #features = self.mhatt(features, features, features, proposal_masks)
        features = self.self_attn[0](features, features, features, attention_weights=dist_weights, way=attention_matrix_way)

        len_nun_max = self.chunk_size
        # len_nun_max = 1 # HACK 1 scene -> 1 description by default

        #objectness_masks = objectness_masks.permute(0, 2, 1).contiguous()  # batch_size, 1, num_proposals
        data_dict["random"] = random.random()

        # copy paste
        feature0 = features.clone()
        if data_dict["istrain"][0] == 1 and data_dict["random"] < 0.5:
            obj_masks = objectness_masks.bool().squeeze(2)  # batch_size, num_proposals
            obj_lens = torch.zeros(batch_size).type_as(feature0).int()
            for i in range(batch_size):
                obj_mask = torch.where(obj_masks[i, :] == True)[0]
                obj_len = obj_mask.shape[0]
                obj_lens[i] = obj_len

            obj_masks_reshape = obj_masks.reshape(batch_size*num_proposal)
            obj_features = features.reshape(batch_size*num_proposal, -1)
            obj_mask = torch.where(obj_masks_reshape[:] == True)[0]
            total_len = obj_mask.shape[0]
            obj_features = obj_features[obj_mask, :].repeat(2,1)  # total_len, hidden_size
            j = 0
            for i in range(batch_size):
                obj_mask = torch.where(obj_masks[i, :] == False)[0]
                obj_len = obj_mask.shape[0]
                j += obj_lens[i]
                if obj_len < total_len - obj_lens[i]:
                    feature0[i, obj_mask, :] = obj_features[j:j + obj_len, :]
                else:
                    feature0[i, obj_mask[:total_len - obj_lens[i]], :] = obj_features[j:j + total_len - obj_lens[i], :]
        
        if use_rl:
            # NOTE visual features must be expanded to matched the number of sampled sentences
            sampled_topn = data_dict["sampled_topn"] # e.g. 3
            feature0 = feature0.unsqueeze(1).repeat(1, sampled_topn, 1, 1).reshape(-1, num_proposal, self.hidden_size)
            dist_weights = dist_weights.unsqueeze(1).repeat(1, sampled_topn, 1, 1, 1).reshape(-1, self.head, num_proposal, num_proposal)

            feature1 = feature0[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(-1, num_proposal, self.hidden_size)
            if dist_weights is not None:
                dist_weights = dist_weights[:, None, :, :, :].repeat(1, len_nun_max, 1, 1, 1).reshape(-1, self.head, num_proposal, num_proposal)

            v_features = feature1

            l_features = data_dict["lang_hiddens"]["sampled"]
            l_masks = data_dict["lang_masks"]["sampled"]
            assert v_features.shape[0] == l_features.shape[0]

            sampled_ref = self.multiplex_attention(v_features, l_features, l_masks, dist_weights, attention_matrix_way)

            with torch.no_grad():
                l_features = data_dict["lang_hiddens"]["baseline"]
                l_masks = data_dict["lang_masks"]["baseline"]
                assert v_features.shape[0] == l_features.shape[0]

                baseline_ref = self.multiplex_attention(v_features, l_features, l_masks, dist_weights, attention_matrix_way)
            
            data_dict["cluster_ref"] = {
                "sampled": sampled_ref,
                "baseline": baseline_ref
            }

        else:
            feature1 = feature0[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(-1, num_proposal, self.hidden_size)
            if dist_weights is not None:
                dist_weights = dist_weights[:, None, :, :, :].repeat(1, len_nun_max, 1, 1, 1).reshape(-1, self.head, num_proposal, num_proposal)

            v_features = feature1

            l_features = data_dict["lang_hiddens"]
            l_masks = data_dict["lang_masks"]
            cluster_ref = self.multiplex_attention(v_features, l_features, l_masks, dist_weights, attention_matrix_way)

            data_dict["cluster_ref"] = cluster_ref

        return data_dict
