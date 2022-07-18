import os
import sys
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LangModule(nn.Module):
    def __init__(self, cfg, 
        emb_size=300, hidden_size=256):
        super().__init__() 

        self.num_text_classes = cfg.model.num_bbox_class
        self.use_lang_classifier = cfg.model.use_lang_classifier
        self.use_bidir = cfg.model.use_bidir

        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=self.use_bidir
        )

        # language classifier
        if self.use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(self.hidden_size, self.num_text_classes),
                nn.Dropout()
            )


    def forward(self, data_dict, use_rl=False):
        """
        encode the input descriptions
        """

        if use_rl:

            sampled_embs = data_dict["lang_feat"]["sampled"]

            # reshape chunks
            batch_size, chunk_size, seq_len, _ = sampled_embs.shape
            sampled_embs = sampled_embs.reshape(-1, seq_len, self.emb_size) # batch_size * chunk_size, seq_len, emb_size

            sampled_len = data_dict["lang_len"]["sampled"].reshape(-1) # batch_size * chunk_size

            # pack sequence
            sampled_feat = pack_padded_sequence(sampled_embs, sampled_len.cpu(), batch_first=True, enforce_sorted=False)

            # encode description
            sampled_hiddens, sampled_last = self.gru(sampled_feat)
            sampled_hiddens, _ = pad_packed_sequence(sampled_hiddens, batch_first=True)
            sampled_last = sampled_last.permute(1, 0, 2).contiguous().flatten(start_dim=1) # batch_size, hidden_size * num_dir

            if self.use_bidir:
                sampled_hiddens = (sampled_hiddens[:, :, :int(sampled_hiddens.shape[-1] / 2)] + sampled_hiddens[:, :, int(sampled_hiddens.shape[-1] / 2):]) / 2
                sampled_last = (sampled_last[:, :int(sampled_last.shape[-1] / 2)] + sampled_last[:, int(sampled_last.shape[-1] / 2):]) / 2

            assert sampled_hiddens.shape[-1] == self.hidden_size
            assert sampled_last.shape[-1] == self.hidden_size

            # HACK zero-padding hiddens
            sampled_pad_hiddens = torch.zeros(batch_size * chunk_size, seq_len, self.hidden_size).type_as(sampled_hiddens)
            sampled_pad_hiddens[:, :sampled_hiddens.shape[1]] = sampled_hiddens

            # sentence mask
            lengths = sampled_len.unsqueeze(1).repeat(1, seq_len) # batch_size * chunk_size, seq_len
            idx = torch.arange(0, seq_len).unsqueeze(0).repeat(lengths.shape[0], 1).type_as(lengths).long() # batch_size * chunk_size, seq_len
            sampled_masks = (idx < lengths).float() # batch_size * chunk_size, seq_len

            # classify
            if self.use_lang_classifier:
                sampled_scores = self.lang_cls(sampled_last)

            with torch.no_grad():
                baseline_embs = data_dict["lang_feat"]["baseline"]

                # reshape chunks
                batch_size, chunk_size, seq_len, _ = baseline_embs.shape
                baseline_embs = baseline_embs.reshape(-1, seq_len, self.emb_size) # batch_size * chunk_size, seq_len, emb_size

                baseline_len = data_dict["lang_len"]["baseline"].reshape(-1) # batch_size * chunk_size

                # pack sequence
                baseline_feat = pack_padded_sequence(baseline_embs, baseline_len.cpu(), batch_first=True, enforce_sorted=False)

                # encode description
                baseline_hiddens, baseline_last = self.gru(baseline_feat)
                baseline_hiddens, _ = pad_packed_sequence(baseline_hiddens, batch_first=True)
                baseline_last = baseline_last.permute(1, 0, 2).contiguous().flatten(start_dim=1) # batch_size, hidden_size * num_dir

                if self.use_bidir:
                    baseline_hiddens = (baseline_hiddens[:, :, :int(baseline_hiddens.shape[-1] / 2)] + baseline_hiddens[:, :, int(baseline_hiddens.shape[-1] / 2):]) / 2
                    baseline_last = (baseline_last[:, :int(baseline_last.shape[-1] / 2)] + baseline_last[:, int(baseline_last.shape[-1] / 2):]) / 2

                assert baseline_hiddens.shape[-1] == self.hidden_size
                assert baseline_last.shape[-1] == self.hidden_size

                # HACK zero-padding hiddens
                baseline_pad_hiddens = torch.zeros(batch_size * chunk_size, seq_len, self.hidden_size).type_as(baseline_hiddens)
                baseline_pad_hiddens[:, :baseline_hiddens.shape[1]] = baseline_hiddens

                # sentence mask
                lengths = baseline_len.unsqueeze(1).repeat(1, seq_len) # batch_size * chunk_size, seq_len
                idx = torch.arange(0, seq_len).unsqueeze(0).repeat(lengths.shape[0], 1).type_as(lengths).long() # batch_size * chunk_size, seq_len
                baseline_masks = (idx < lengths).float() # batch_size * chunk_size, seq_len

                # classify
                if self.use_lang_classifier:
                    baseline_scores = self.lang_cls(baseline_last)

            # store
            data_dict["lang_masks"] = {
                "sampled": sampled_masks, # B, T
                "baseline": baseline_masks # B, T
            }

            # store the encoded language features
            data_dict["lang_hiddens"] = {
                "sampled": sampled_pad_hiddens, # B, T, hidden_size
                "baseline": baseline_pad_hiddens, # B, T, hidden_size
            }

            data_dict["lang_emb"] = {
                "sampled": sampled_last, # B, hidden_size
                "baseline": baseline_last # B, hidden_size
            }

            data_dict["lang_scores"] = {
                "sampled": sampled_scores,
                "baseline": baseline_scores
            }

        else:

            word_embs = data_dict["lang_feat"] # batch_size, chunk_size, seq_len, emb_size    
            batch_size, chunk_size, seq_len, _ = word_embs.shape
            word_embs = word_embs.reshape(-1, seq_len, self.emb_size) # batch_size * chunk_size, seq_len, emb_size

            lang_len = data_dict["lang_len"] # batch_size, chunk_size
            lang_len = lang_len.reshape(-1) # batch_size * chunk_size

            lang_feat = pack_padded_sequence(word_embs, lang_len.cpu(), batch_first=True, enforce_sorted=False)
        
            # encode description
            lang_hiddens, lang_last = self.gru(lang_feat)
            lang_hiddens, _ = pad_packed_sequence(lang_hiddens, batch_first=True)
            lang_last = lang_last.permute(1, 0, 2).contiguous().flatten(start_dim=1) # batch_size * chunk_size, hidden_size * num_dir

            if self.use_bidir:
                lang_hiddens = (lang_hiddens[:, :, :int(lang_hiddens.shape[-1] / 2)] + lang_hiddens[:, :, int(lang_hiddens.shape[-1] / 2):]) / 2
                lang_last = (lang_last[:, :int(lang_last.shape[-1] / 2)] + lang_last[:, int(lang_last.shape[-1] / 2):]) / 2
            
            assert lang_hiddens.shape[-1] == self.hidden_size
            assert lang_last.shape[-1] == self.hidden_size

            # HACK zero-padding hiddens
            pad_hiddens = torch.zeros(batch_size * chunk_size, seq_len, self.hidden_size).type_as(lang_hiddens)
            pad_hiddens[:, :lang_hiddens.shape[1]] = lang_hiddens

            # sentence mask
            lengths = lang_len.unsqueeze(1).repeat(1, seq_len) # batch_size * chunk_size, seq_len
            idx = torch.arange(0, seq_len).unsqueeze(0).repeat(lengths.shape[0], 1).type_as(lengths).long() # batch_size * chunk_size, seq_len
            lang_masks = (idx < lengths).float() # batch_size * chunk_size, seq_len
            data_dict["lang_masks"] = lang_masks # batch_size * chunk_size, seq_len

            # store the encoded language features
            data_dict["lang_hiddens"] = pad_hiddens # batch_size * chunk_size, seq_len, hidden_size
            data_dict["lang_emb"] = lang_last # batch_size * chunk_size, hidden_size
            
            # classify
            if self.use_lang_classifier:
                data_dict["lang_scores"] = self.lang_cls(data_dict["lang_emb"]) # batch_size * chunk_size, num_text_classes

        return data_dict

