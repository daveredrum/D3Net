import os
import sys

import torch.nn as nn

from model.lang_module import LangModule
from model.match_module import MatchModule, TransformerMatchModule


class ListenerNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.match_type = cfg.model.match_type

        # --------- LANGUAGE ENCODING ---------
        # Encode the input descriptions into vectors
        # (including attention and language classification)
        self.lang = LangModule(cfg)

        # --------- PROPOSAL MATCHING ---------
        # Match the generated proposals and select the most confident ones
        if self.match_type == "ScanRefer":
            # # ********* ScanRefer Matching *********
            self.match = MatchModule(cfg)
        elif self.match_type == "Transformer":
            # # ********* Transformer Matching *********
            self.match = TransformerMatchModule(cfg)
        else:
            raise NotImplementedError("Matching type not supported.")

    def forward(self, data_dict, use_rl=False):

        #######################################
        #                                     #
        #           LANGUAGE BRANCH           #
        #                                     #
        #######################################

        # --------- LANGUAGE ENCODING ---------
        data_dict = self.lang(data_dict, use_rl)

        #######################################
        #                                     #
        #          PROPOSAL MATCHING          #
        #                                     #
        #######################################

        # --------- PROPOSAL MATCHING ---------
        data_dict = self.match(data_dict, use_rl)

        return data_dict
