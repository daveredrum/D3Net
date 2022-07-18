
import torch

import torch.nn as nn

from data.scannet.model_util_scannet import ScannetDatasetConfig

from model.graph_module import GraphModule
from model.caption_module import TopDownSceneCaptionModule

class SpeakerNet(nn.Module):
    def __init__(self, cfg, vocabulary, embeddings):
        super().__init__()

        self.cfg = cfg
        self.vocabulary = vocabulary
        self.embeddings = embeddings

        if self.cfg.model.num_graph_steps > 0:
            self.graph = GraphModule(self.cfg.model.m, 128, self.cfg.model.num_graph_steps, 
                self.cfg.model.max_num_proposal, 128, self.cfg.model.num_locals, 
                return_edge=self.cfg.model.use_relation, return_orientation=self.cfg.model.use_orientation)

        # Caption generation
        if not self.cfg.model.no_captioning:
            self.caption = TopDownSceneCaptionModule(cfg, vocabulary, embeddings, 
                num_proposals=self.cfg.model.max_num_proposal, num_locals=self.cfg.model.num_locals, 
                use_relation=self.cfg.model.use_relation, use_oracle=self.cfg.model.no_detection)

    # NOTE direct access only during inference
    def forward(self, data_dict, use_tf=True, use_rl=False, is_eval=False, beam_opt={}):

        #######################################
        #                                     #
        #           GRAPH ENHANCEMENT         #
        #                                     #
        #######################################

        if self.cfg.model.num_graph_steps > 0: 
            data_dict = self.graph(data_dict)

        #######################################
        #                                     #
        #            CAPTION BRANCH           #
        #                                     #
        #######################################

        # --------- CAPTION GENERATION ---------
        if not self.cfg.model.no_captioning:
            data_dict = self.caption(data_dict, use_tf, use_rl, is_eval, beam_opt)

        return data_dict
