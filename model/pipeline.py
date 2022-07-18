from copy import deepcopy
import os
import sys
import torch
import random

import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from data.scannet.model_util_scannet import ScannetDatasetConfig

from model.pointgroup import PointGroup
from model.speaker import SpeakerNet
from model.listener import ListenerNet

from lib.det.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.captioning.loss_helper import get_loss as get_captioning_loss
from lib.captioning.eval_helper import eval_caption_step, eval_caption_epoch
from lib.grounding.loss_helper import get_loss as get_grounding_loss
from lib.grounding.eval_helper import get_eval as get_grounding_eval


class PipelineNet(pl.LightningModule):
    def __init__(self, cfg, dataset=None):
        super().__init__()

        self.cfg = cfg

        self.init_random_seed()

        self.no_detection = cfg.model.no_detection
        self.no_captioning = cfg.model.no_captioning
        self.no_grounding = cfg.model.no_grounding
        self._get_current_mode()

        if dataset:
            self.vocabulary = dataset["train"].vocabulary
            self.register_buffer("embeddings", torch.FloatTensor(dataset["train"].glove))
            self.dataset_chunk_data = dataset["val"].chunked_data

            self.beam_opt = {
                "train_beam_size": self.cfg.train.beam_size,
                "train_sample_topn": self.cfg.train.sample_topn,
                "eval_beam_size": self.cfg.train.beam_size
            }
            self.loss_opt = {
                "use_rl": self.cfg.train.use_rl,
                "sample_topn": self.cfg.train.sample_topn,
                "idx2word": self.vocabulary["idx2word"],
                "train_dataset_data": dataset["train"].chunked_data,
                "organized_data": dataset["train"].organized,
                "max_len": self.cfg.data.max_spk_len + 2,
                "loss_type": self.cfg.model.loss_type,

                "ref_reward_weight": self.cfg.train.ref_reward_weight,
                "lang_reward_weight": self.cfg.train.lang_reward_weight,
                "listener_reward_weight": self.cfg.train.listener_reward_weight,
                "caption_reward_weight": self.cfg.train.caption_reward_weight,
            }

        if not self.no_detection:
            self.detector = PointGroup(cfg)

        if not self.no_captioning:
            self.speaker = SpeakerNet(cfg, self.vocabulary, self.embeddings)

        if not self.no_grounding:
            self.listener = ListenerNet(cfg)

        self.DC = ScannetDatasetConfig(cfg)

        self.num_class = cfg.model.num_bbox_class
        self.num_proposal = cfg.model.max_num_proposal

        self.use_lang_classifier = cfg.model.use_lang_classifier

        self.post_dict = {
            "remove_empty_box": False, 
            "use_3d_nms": True, 
            "nms_iou": 0.25,
            "use_old_type_nms": False, 
            "cls_nms": True, 
            "per_class_proposal": True,
            "conf_thresh": 0.09,
            "dataset_config": self.DC
        }
        self.ap_calculator = APCalculator(0.5, self.DC.class2type)

    def _get_current_mode(self):
        """
            different modes of the pipeline:
                0: detector
                1: detector -> speaker
                2: detector -> listener
                3: detector -> speaker -> listener
                4: GT -> speaker
                5: GT -> listener
                6: GT -> speaker -> listener
        """

        assert not self.no_detection or not self.no_captioning or not self.no_grounding, \
            "invalid mode, detection: {}, captioning: {}, grounding: {}".format(
                not self.no_detection, not self.no_captioning, not self.no_grounding
            )
        
        if self.no_detection:
            if self.no_grounding and not self.no_captioning:
                self.mode = 4
            elif not self.no_grounding and self.no_captioning:
                self.mode = 5
            else:
                self.mode = 6
        else:
            if self.no_grounding and self.no_captioning:
                self.mode = 0
            elif self.no_grounding and not self.no_captioning:
                self.mode = 1
            elif not self.no_grounding and self.no_captioning:
                self.mode = 2
            else:
                self.mode = 3

        
    def init_random_seed(self):
        print("=> setting random seed...")
        if self.cfg.general.manual_seed:
            random.seed(self.cfg.general.manual_seed)
            np.random.seed(self.cfg.general.manual_seed)
            torch.manual_seed(self.cfg.general.manual_seed)
            torch.cuda.manual_seed_all(self.cfg.general.manual_seed)

    def training_step(self, data_dict, idx):
        if self.global_step % self.cfg.model.clear_cache_steps == 0:
            torch.cuda.empty_cache()

        if self.mode == 0: # detector
            # forward pass
            data_dict = self.detector.feed(data_dict, self.current_epoch)
            _, data_dict = self.detector.parse_feed_ret(data_dict, self.current_epoch)
            data_dict = self.detector.loss(data_dict, self.current_epoch)

            loss = data_dict["total_loss"][0]

            # log
            in_prog_bar = ["total_loss"]
            for key, value in data_dict.items():
                if "loss" in key:
                    self.log("train/{}".format(key), value[0], prog_bar=key in in_prog_bar, on_step=True, on_epoch=True, sync_dist=True)

        elif self.mode == 1: # detector -> speaker
            # forward pass
            data_dict = self.detector.feed(data_dict, self.current_epoch)
            _, data_dict = self.detector.parse_feed_ret(data_dict)
            data_dict = self.detector.loss(data_dict, self.current_epoch)
            data_dict = self.speaker(data_dict)

            _, data_dict = get_captioning_loss(
                data_dict,
                caption=not self.cfg.model.no_captioning,
                orientation=self.cfg.model.use_orientation,
                num_bins=self.cfg.data.num_ori_bins,
                loss_opt=self.loss_opt
            )

            loss = data_dict["total_loss"][0] + data_dict["cap_loss"] + 0.1 * data_dict["ori_loss"]

            # unpack
            log_dict = {
                "loss": loss,
                "detect_loss": data_dict["total_loss"][0],
                "captioning_loss": data_dict["cap_loss"],
                "orientation_loss": data_dict["ori_loss"],

                "cap_acc": data_dict["cap_acc"],
                "ori_acc": data_dict["ori_acc"],
                "pred_ious": data_dict["pred_ious"],
            }

            # log
            in_prog_bar = ["cap_acc"]
            for key, value in log_dict.items():
                ctg = "loss" if "loss" in key else "score"
                self.log("train_{}/{}".format(ctg, key), value, prog_bar=(key in in_prog_bar), on_step=True, on_epoch=True, sync_dist=True)

        elif self.mode == 2: # detector -> listener
            # forward pass
            data_dict = self.detector.feed(data_dict, self.current_epoch)
            _, data_dict = self.detector.parse_feed_ret(data_dict)
            data_dict = self.detector.loss(data_dict, self.current_epoch)
            data_dict = self.listener(data_dict)

            _, data_dict = get_grounding_loss(
                data_dict,
                use_oracle=self.no_detection,
                grounding=not self.no_grounding,
                use_lang_classifier=self.use_lang_classifier,
                use_rl=False
            )

            loss = data_dict["total_loss"][0] + data_dict["ref_loss"] + data_dict["lang_loss"]

            # unpack
            log_dict = {
                "loss": loss,

                "detect_loss": data_dict["total_loss"][0],
                "grounding_loss": data_dict["ref_loss"],
                "lobjcls_loss": data_dict["lang_loss"],

                "ref_acc_mean": data_dict["ref_acc_mean"],
                "ref_iou_mean": data_dict["ref_iou_mean"],
                "best_ious_mean": data_dict["best_ious_mean"],

                "ref_iou_rate_0.25": data_dict["ref_iou_rate_0.25"],
                "ref_iou_rate_0.5": data_dict["ref_iou_rate_0.5"],

                "lang_acc": data_dict["lang_acc"],
            }

            # log
            in_prog_bar = ["ref_iou_rate_0.5"]
            for key, value in log_dict.items():
                ctg = "loss" if "loss" in key else "score"
                self.log("train_{}/{}".format(ctg, key), value, prog_bar=(key in in_prog_bar), on_step=True, on_epoch=True, sync_dist=True)

        elif self.mode == 3: # detector -> speaker -> listener
            assert len(data_dict) == 2
            # ---------- for SpeakerDataset ----------------
            spk_data_dict = data_dict[0]

            # forward pass
            spk_data_dict = self.detector.feed(spk_data_dict, self.current_epoch)
            _, spk_data_dict = self.detector.parse_feed_ret(spk_data_dict)
            spk_data_dict = self.detector.loss(spk_data_dict, self.current_epoch)
            spk_data_dict = self.speaker(spk_data_dict, use_rl=self.cfg.train.use_rl, is_eval=False, beam_opt=self.beam_opt)
            spk_data_dict = self.moderator(spk_data_dict, self.cfg.data.max_spk_len + 2)
            spk_data_dict = self.listener(spk_data_dict, use_rl=self.cfg.train.use_rl)
            _, spk_data_dict = get_grounding_loss(
                spk_data_dict,
                use_oracle=self.no_detection,
                grounding=not self.no_grounding,
                use_lang_classifier=self.use_lang_classifier,
                use_rl=self.cfg.train.use_rl
            )
            _, spk_data_dict = get_captioning_loss(
                spk_data_dict,
                caption=not self.cfg.model.no_captioning,
                orientation=self.cfg.model.use_orientation,
                num_bins=self.cfg.data.num_ori_bins,
                loss_opt=self.loss_opt
            )

            loss = spk_data_dict["total_loss"][0] + spk_data_dict["cap_loss"] + 0.1 * spk_data_dict["ori_loss"]
            loss += spk_data_dict["ref_loss"] + spk_data_dict["lang_loss"]

            # ---------- for ListenerDataset ----------------
            lis_data_dict = data_dict[1]

            # forward pass
            lis_data_dict = self.detector.feed(lis_data_dict, self.current_epoch)
            _, lis_data_dict = self.detector.parse_feed_ret(lis_data_dict)
            lis_data_dict = self.detector.loss(lis_data_dict, self.current_epoch)
            lis_data_dict = self.listener(lis_data_dict)
            _, lis_data_dict = get_grounding_loss(
                lis_data_dict,
                use_oracle=self.no_detection,
                grounding=not self.no_grounding,
                use_lang_classifier=self.use_lang_classifier,
                use_rl=False
            )

            loss += lis_data_dict["total_loss"][0] + lis_data_dict["ref_loss"] + lis_data_dict["lang_loss"]

            # unpack
            log_dict = {
                "loss": loss,

                "detect_loss": (spk_data_dict["total_loss"][0] + lis_data_dict["total_loss"][0]) / 2,
                "captioning_loss": spk_data_dict["cap_loss"],
                "orientation_loss": spk_data_dict["ori_loss"],

                "grounding_loss": (spk_data_dict["ref_loss"] + lis_data_dict["ref_loss"]) / 2,
                "lobjcls_loss": (spk_data_dict["lang_loss"] + lis_data_dict["lang_loss"]) / 2,

                "cap_acc": spk_data_dict["cap_acc"],
                "ori_acc": spk_data_dict["ori_acc"],
                "pred_ious": spk_data_dict["pred_ious"],

                "cap_rwd": spk_data_dict["cap_rwd"],
                "loc_rwd": spk_data_dict["loc_rwd"],
                "ttl_rwd": spk_data_dict["ttl_rwd"],

                "ref_acc_mean": (spk_data_dict["ref_acc_mean"] + lis_data_dict["ref_acc_mean"]) / 2,
                "ref_iou_mean": (spk_data_dict["ref_iou_mean"] + lis_data_dict["ref_iou_mean"]) / 2,
                "best_ious_mean": (spk_data_dict["best_ious_mean"] + lis_data_dict["best_ious_mean"]) / 2,

                "ref_iou_rate_0.25": (spk_data_dict["ref_iou_rate_0.25"] + lis_data_dict["ref_iou_rate_0.25"]) / 2,
                "ref_iou_rate_0.5": (spk_data_dict["ref_iou_rate_0.5"] + lis_data_dict["ref_iou_rate_0.5"]) / 2,

                "lang_acc": (spk_data_dict["lang_acc"] + lis_data_dict["lang_acc"]) / 2,
            }

            # log
            in_prog_bar = ["ref_iou_rate_0.5"]
            for key, value in log_dict.items():
                ctg = "loss" if "loss" in key else "score"
                self.log("train_{}/{}".format(ctg, key), value, prog_bar=(key in in_prog_bar), on_step=True, on_epoch=True, sync_dist=True)

        elif self.mode == 4: # GT -> speaker
            # forward pass
            data_dict = self.speaker(data_dict)

            _, data_dict = get_captioning_loss(
                data_dict,
                caption=not self.cfg.model.no_captioning,
                orientation=self.cfg.model.use_orientation,
                num_bins=self.cfg.data.num_ori_bins,
                loss_opt=self.loss_opt
            )

            loss = data_dict["cap_loss"] + 0.1 * data_dict["ori_loss"]

            # unpack
            log_dict = {
                "loss": loss,
                "captioning_loss": data_dict["cap_loss"],
                "orientation_loss": data_dict["ori_loss"],

                "cap_acc": data_dict["cap_acc"],
                "ori_acc": data_dict["ori_acc"],
                "pred_ious": data_dict["pred_ious"],
            }

            # log
            in_prog_bar = ["cap_acc"]
            for key, value in log_dict.items():
                ctg = "loss" if "loss" in key else "score"
                self.log("train_{}/{}".format(ctg, key), value, prog_bar=(key in in_prog_bar), on_step=True, on_epoch=True, sync_dist=True)

        elif self.mode == 5: # GT -> listener
            # forward pass
            data_dict = self.listener(data_dict)

            _, data_dict = get_grounding_loss(
                data_dict,
                use_oracle=self.no_detection,
                grounding=not self.no_grounding,
                use_lang_classifier=self.use_lang_classifier,
                use_rl=False
            )

            loss = data_dict["ref_loss"] + data_dict["lang_loss"]

            # unpack
            log_dict = {
                "loss": loss,

                "grounding_loss": data_dict["ref_loss"],
                "lobjcls_loss": data_dict["lang_loss"],

                "ref_acc_mean": data_dict["ref_acc_mean"],
                "ref_iou_mean": data_dict["ref_iou_mean"],
                "best_ious_mean": data_dict["best_ious_mean"],

                "ref_iou_rate_0.25": data_dict["ref_iou_rate_0.25"],
                "ref_iou_rate_0.5": data_dict["ref_iou_rate_0.5"],

                "lang_acc": data_dict["lang_acc"],
            }

            # log
            in_prog_bar = ["ref_iou_rate_0.5"]
            for key, value in log_dict.items():
                ctg = "loss" if "loss" in key else "score"
                self.log("train_{}/{}".format(ctg, key), value, prog_bar=(key in in_prog_bar), on_step=True, on_epoch=True, sync_dist=True)

        elif self.mode == 6: # GT -> speaker -> listener
            assert len(data_dict) == 2
            # ---------- for SpeakerDataset ----------------
            spk_data_dict = data_dict[0]

            # forward pass
            spk_data_dict = self.speaker(spk_data_dict, use_rl=self.cfg.train.use_rl, is_eval=False, beam_opt=self.beam_opt)
            spk_data_dict = self.moderator(spk_data_dict, self.cfg.data.max_spk_len + 2)
            spk_data_dict = self.listener(spk_data_dict, use_rl=self.cfg.train.use_rl)
            _, spk_data_dict = get_grounding_loss(
                spk_data_dict,
                use_oracle=self.no_detection,
                grounding=not self.no_grounding,
                use_lang_classifier=self.use_lang_classifier,
                use_rl=self.cfg.train.use_rl
            )
            _, spk_data_dict = get_captioning_loss(
                spk_data_dict,
                caption=not self.cfg.model.no_captioning,
                orientation=self.cfg.model.use_orientation,
                num_bins=self.cfg.data.num_ori_bins,
                loss_opt=self.loss_opt
            )

            loss = spk_data_dict["cap_loss"] + 0.1 * spk_data_dict["ori_loss"]
            loss += spk_data_dict["ref_loss"] + spk_data_dict["lang_loss"]

            # ---------- for ListenerDataset ----------------
            lis_data_dict = data_dict[1]

            # forward pass
            lis_data_dict = self.listener(lis_data_dict)
            _, lis_data_dict = get_grounding_loss(
                lis_data_dict,
                use_oracle=self.no_detection,
                grounding=not self.no_grounding,
                use_lang_classifier=self.use_lang_classifier,
                use_rl=False
            )

            loss += lis_data_dict["ref_loss"] + lis_data_dict["lang_loss"]

            # unpack
            log_dict = {
                "loss": loss,

                "captioning_loss": spk_data_dict["cap_loss"],
                "orientation_loss": spk_data_dict["ori_loss"],

                "grounding_loss": (spk_data_dict["ref_loss"] + lis_data_dict["ref_loss"]) / 2,
                "lobjcls_loss": (spk_data_dict["lang_loss"] + lis_data_dict["lang_loss"]) / 2,

                "cap_acc": spk_data_dict["cap_acc"],
                "ori_acc": spk_data_dict["ori_acc"],
                "pred_ious": spk_data_dict["pred_ious"],

                "cap_rwd": spk_data_dict["cap_rwd"],
                "loc_rwd": spk_data_dict["loc_rwd"],
                "ttl_rwd": spk_data_dict["ttl_rwd"],

                "ref_acc_mean": (spk_data_dict["ref_acc_mean"] + lis_data_dict["ref_acc_mean"]) / 2,
                "ref_iou_mean": (spk_data_dict["ref_iou_mean"] + lis_data_dict["ref_iou_mean"]) / 2,
                "best_ious_mean": (spk_data_dict["best_ious_mean"] + lis_data_dict["best_ious_mean"]) / 2,

                "ref_iou_rate_0.25": (spk_data_dict["ref_iou_rate_0.25"] + lis_data_dict["ref_iou_rate_0.25"]) / 2,
                "ref_iou_rate_0.5": (spk_data_dict["ref_iou_rate_0.5"] + lis_data_dict["ref_iou_rate_0.5"]) / 2,

                "lang_acc": (spk_data_dict["lang_acc"] + lis_data_dict["lang_acc"]) / 2,
            }

            # log
            in_prog_bar = ["ref_iou_rate_0.5"]
            for key, value in log_dict.items():
                ctg = "loss" if "loss" in key else "score"
                self.log("train_{}/{}".format(ctg, key), value, prog_bar=(key in in_prog_bar), on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, data_dict, idx, dataloader_idx=0):
        if self.global_step % self.cfg.model.clear_cache_steps == 0:
            torch.cuda.empty_cache()

        if self.mode == 0:
            data_dict = self.detector.feed(data_dict, self.current_epoch)
            _, data_dict = self.detector.parse_feed_ret(data_dict, self.current_epoch)
            data_dict = self.detector.loss(data_dict, self.current_epoch)
            
            for key, value in data_dict.items():
                if "loss" in key:
                    self.log("val_loss/{}".format(key), value[0], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        elif self.mode == 1:
            data_dict = self.detector.feed(data_dict, self.current_epoch)
            _, data_dict = self.detector.parse_feed_ret(data_dict)
            data_dict = self.detector.loss(data_dict, self.current_epoch)

            data_dict = self.speaker(data_dict, use_tf=False, use_rl=False, is_eval=True, beam_opt=self.beam_opt)

            # eval speaker
            candidates = eval_caption_step(
                cfg=self.cfg,
                data_dict=data_dict,
                dataset_chunked_data=self.dataset_chunk_data,
                dataset_vocabulary=self.vocabulary
            )

            return candidates

        elif self.mode == 2:
            data_dict = self.detector.feed(data_dict, self.current_epoch)
            _, data_dict = self.detector.parse_feed_ret(data_dict)
            data_dict = self.detector.loss(data_dict, self.current_epoch)

            data_dict = self.listener(data_dict)

            _, data_dict = get_grounding_loss(
                data_dict,
                use_oracle=self.no_detection,
                grounding=not self.no_grounding,
                use_lang_classifier=self.use_lang_classifier,
                use_rl=False
            )

            log_dict = {
                "ref_acc_mean": data_dict["ref_acc_mean"],
                "ref_iou_mean": data_dict["ref_iou_mean"],
                "best_ious_mean": data_dict["best_ious_mean"],

                "ref_iou_rate_0.25": data_dict["ref_iou_rate_0.25"],
                "ref_iou_rate_0.5": data_dict["ref_iou_rate_0.5"],

                "lang_acc": data_dict["lang_acc"],
            }

            # log
            for key, value in log_dict.items():
                ctg = "loss" if "loss" in key else "score"
                self.log("val_{}/{}".format(ctg, key), value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        elif self.mode == 3:
            data_dict = self.detector.feed(data_dict, self.current_epoch)
            _, data_dict = self.detector.parse_feed_ret(data_dict)
            data_dict = self.detector.loss(data_dict, self.current_epoch)

            if dataloader_idx == 0:
                data_dict = self.speaker(data_dict, use_tf=False, use_rl=False, is_eval=True, beam_opt=self.beam_opt)

                # eval speaker
                candidates = eval_caption_step(
                    cfg=self.cfg,
                    data_dict=data_dict,
                    dataset_chunked_data=self.dataset_chunk_data,
                    dataset_vocabulary=self.vocabulary
                )

                return candidates

            elif dataloader_idx == 1:
                data_dict = self.listener(data_dict)

                _, data_dict = get_grounding_loss(
                    data_dict,
                    use_oracle=self.no_detection,
                    grounding=not self.no_grounding,
                    use_lang_classifier=self.use_lang_classifier,
                    use_rl=False
                )

                log_dict = {
                    "ref_acc_mean": data_dict["ref_acc_mean"],
                    "ref_iou_mean": data_dict["ref_iou_mean"],
                    "best_ious_mean": data_dict["best_ious_mean"],

                    "ref_iou_rate_0.25": data_dict["ref_iou_rate_0.25"],
                    "ref_iou_rate_0.5": data_dict["ref_iou_rate_0.5"],

                    "lang_acc": data_dict["lang_acc"],
                }
            
                return log_dict

            else:
                raise NotImplementedError()

        elif self.mode == 4:
            data_dict = self.speaker(data_dict, use_tf=False, use_rl=False, is_eval=True, beam_opt=self.beam_opt)

            # eval speaker
            candidates = eval_caption_step(
                cfg=self.cfg,
                data_dict=data_dict,
                dataset_chunked_data=self.dataset_chunk_data,
                dataset_vocabulary=self.vocabulary
            )

            return candidates

        elif self.mode == 5:
            data_dict = self.listener(data_dict)

            _, data_dict = get_grounding_loss(
                data_dict,
                use_oracle=self.no_detection,
                grounding=not self.no_grounding,
                use_lang_classifier=self.use_lang_classifier,
                use_rl=False
            )

            log_dict = {
                "ref_acc_mean": data_dict["ref_acc_mean"],
                "ref_iou_mean": data_dict["ref_iou_mean"],
                "best_ious_mean": data_dict["best_ious_mean"],

                "ref_iou_rate_0.25": data_dict["ref_iou_rate_0.25"],
                "ref_iou_rate_0.5": data_dict["ref_iou_rate_0.5"],

                "lang_acc": data_dict["lang_acc"],
            }

            # log
            for key, value in log_dict.items():
                ctg = "loss" if "loss" in key else "score"
                self.log("val_{}/{}".format(ctg, key), value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)


        elif self.mode == 6:
            if dataloader_idx == 0:
                data_dict = self.speaker(data_dict, use_tf=False, use_rl=False, is_eval=True, beam_opt=self.beam_opt)

                # eval speaker
                candidates = eval_caption_step(
                    cfg=self.cfg,
                    data_dict=data_dict,
                    dataset_chunked_data=self.dataset_chunk_data,
                    dataset_vocabulary=self.vocabulary
                )

                return candidates

            elif dataloader_idx == 1:
                data_dict = self.listener(data_dict)

                _, data_dict = get_grounding_loss(
                    data_dict,
                    use_oracle=self.no_detection,
                    grounding=not self.no_grounding,
                    use_lang_classifier=self.use_lang_classifier,
                    use_rl=False
                )

                log_dict = {
                    "ref_acc_mean": data_dict["ref_acc_mean"],
                    "ref_iou_mean": data_dict["ref_iou_mean"],
                    "best_ious_mean": data_dict["best_ious_mean"],

                    "ref_iou_rate_0.25": data_dict["ref_iou_rate_0.25"],
                    "ref_iou_rate_0.5": data_dict["ref_iou_rate_0.5"],

                    "lang_acc": data_dict["lang_acc"],
                }
            
                return log_dict

            else:
                raise NotImplementedError()

    def validation_epoch_end(self, outputs):
        log_dict = {}
        
        if self.mode == 0:
            pass

        elif self.mode == 1 or self.mode == 4:
            # aggregate captioning outputs
            candidates = {}
            for outs in outputs:
                for key, value in outs.items():
                    if key not in candidates:
                        candidates[key] = value

            # evaluate captions
            bleu, cider, rouge, meteor = eval_caption_epoch(
                candidates=candidates,
                cfg=self.cfg,
                device=self.device,
                phase="val",
                force=True,
                max_len=self.cfg.eval.max_des_len + 2,
                min_iou=self.cfg.eval.min_iou_threshold
            )

            log_dict = {
                "bleu-1": bleu[0][0],
                "bleu-2": bleu[0][1],
                "bleu-3": bleu[0][2],
                "bleu-4": bleu[0][3],
                "cider": cider[0],
                "meteor": meteor[0],
                "rouge": rouge[0]
            }

            # log
            for key, value in log_dict.items():
                self.log("val_score/{}".format(key), value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        elif self.mode == 2 or self.mode == 5:
            pass

        elif self.mode == 3 or self.mode == 6:
            # aggregate captioning outputs
            candidates = {}
            for outs in outputs[0]:
                for key, value in outs.items():
                    if key not in candidates:
                        candidates[key] = value

            # evaluate captions
            bleu, cider, rouge, meteor = eval_caption_epoch(
                candidates=candidates,
                cfg=self.cfg,
                device=self.device,
                phase="val",
                max_len=self.cfg.eval.max_des_len + 2,
                min_iou=self.cfg.eval.min_iou_threshold
            )

            log_dict = {
                "bleu-1": bleu[0][0],
                "bleu-2": bleu[0][1],
                "bleu-3": bleu[0][2],
                "bleu-4": bleu[0][3],
                "cider": cider[0],
                "meteor": meteor[0],
                "rouge": rouge[0]
            }

            # aggregate grounding metrics
            metrics = {}
            for outs in outputs[1]:
                for key, value in outs.items():
                    if key not in metrics: metrics[key] = []
                    
                    if isinstance(value, torch.Tensor):
                        metrics[key].append(value.item())
                    else:
                        metrics[key].append(value)

                    # metrics[key].append(value)

            for key, value in metrics.items():
                log_dict[key] = np.mean(value)

            log_dict["combined"] = log_dict["cider"] + log_dict["ref_iou_rate_0.5"]

            # log
            for key, value in log_dict.items():
                self.log("val_score/{}".format(key), value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)


    def configure_optimizers(self):
        print("=> configure optimizer...")

        optim_class_name = self.cfg.train.optim.classname
        optim = getattr(torch.optim, optim_class_name)
        if optim_class_name == "Adam" or optim_class_name == "AdamW":
            optimizer = optim(filter(lambda p: p.requires_grad, self.parameters()), lr=self.cfg.train.optim.lr, weight_decay=self.cfg.train.optim.weight_decay)
        elif optim_class_name == "SGD":
            optimizer = optim(filter(lambda p: p.requires_grad, self.parameters()), lr=self.cfg.train.optim.lr, momentum=self.cfg.train.optim.momentum, weight_decay=self.cfg.train.optim.weight_decay)
        else:
            raise NotImplemented

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.train.epochs)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=10,
            gamma=0.8
        )

        return [optimizer], [scheduler]

    def moderator(self, data_dict, max_spk_len):
        """ decode the speaker output captions and create
            differentiable input for the listener

        Args:
            speaker_data ([type]): [description]

        Returns:
            [type]: [description]
        """
    
        bbox_features = data_dict["bbox_feature"]
        des_sampled = data_dict["lang_cap"] # [batch_size...[sample_topn...[seq_len...]]]
        des_baseline = data_dict["baseline_cap"] # [batch_size...[sample_topn...[seq_len...]]]
        
        # aggregate the samples into batches so that the new batch_size is batch_size * sample_topn
        assert len(des_sampled[0]) == len(des_baseline[0])
        batch_size, sample_topn = len(des_sampled), len(des_sampled[0])
        chunk_size = batch_size // bbox_features.shape[0]
        

        des_sampled_mat = torch.zeros(batch_size, sample_topn, max_spk_len).type_as(bbox_features).long()
        des_sampled_lens = torch.zeros(batch_size, sample_topn).type_as(bbox_features).long()
        des_baseline_mat = torch.zeros(batch_size, sample_topn, max_spk_len).type_as(bbox_features).long()
        des_baseline_lens = torch.zeros(batch_size, sample_topn).type_as(bbox_features).long()
        for batch_id in range(batch_size):
            for sample_id in range(sample_topn):
                sos_idx, eos_idx = 2, 3
                sos_idx = torch.Tensor([sos_idx]).type_as(bbox_features)
                eos_idx = torch.Tensor([eos_idx]).type_as(bbox_features)

                # for sampled descriptions
                sampled = des_sampled[batch_id][sample_id]
                sampled = torch.cat([sos_idx, sampled])
                if (sampled == eos_idx).sum() == 0: 
                    sampled = torch.cat([sampled, eos_idx])

                assert sampled.shape[0] <= des_sampled_mat.shape[-1]
            
                # store
                sampled_len = sampled.shape[0]
                des_sampled_mat[batch_id, sample_id, :sampled_len] = sampled
                des_sampled_lens[batch_id, sample_id] = sampled_len

                # for baseline descriptions
                baseline = des_baseline[batch_id][sample_id]
                baseline = torch.cat([sos_idx, baseline])
                if (baseline == eos_idx).sum() == 0: 
                    baseline = torch.cat([baseline, eos_idx])

                assert baseline.shape[0] <= des_baseline_mat.shape[-1]

                # store
                baseline_len = baseline.shape[0]
                des_baseline_mat[batch_id, sample_id, :baseline_len] = baseline
                des_baseline_lens[batch_id, sample_id] = baseline_len


        vocab_size, emb_dim = self.embeddings.shape

        # convert sampled descriptions to word embeddings
        des_sampled_onehot = torch.zeros(batch_size, sample_topn, max_spk_len, vocab_size).type_as(bbox_features)
        des_sampled_onehot.scatter_(-1, des_sampled_mat.unsqueeze(-1).repeat(1, 1, 1, vocab_size), 1) # batch_size, sample_topn, num_words - 1/max_len, vocab_size
        des_sampled_embs = torch.matmul(des_sampled_onehot, self.embeddings.view(1, 1, vocab_size, emb_dim).repeat(batch_size, sample_topn, 1, 1)) # batch_size, sample_topn, num_words - 1/max_len, 300

        # convert baseline descriptions to word embeddings
        des_baseline_onehot = torch.zeros(batch_size, sample_topn, max_spk_len, vocab_size).type_as(bbox_features)
        des_baseline_onehot.scatter_(-1, des_baseline_mat.unsqueeze(-1).repeat(1, 1, 1, vocab_size), 1) # batch_size, num_words - 1/max_len, vocab_size
        des_baseline_embs = torch.matmul(des_baseline_onehot, self.embeddings.view(1, 1, vocab_size, emb_dim).repeat(batch_size, sample_topn, 1, 1)) # batch_size, num_words - 1/max_len, 300

        # shift the embedding matrix and expose the chunk dim
        des_sampled_embs = des_sampled_embs.reshape(-1, chunk_size, sample_topn, max_spk_len, 300)
        des_sampled_embs = des_sampled_embs.transpose(2, 1).reshape(-1, chunk_size, max_spk_len, 300)
        des_baseline_embs = des_baseline_embs.reshape(-1, chunk_size, sample_topn, max_spk_len, 300)
        des_baseline_embs = des_baseline_embs.transpose(2, 1).reshape(-1, chunk_size, max_spk_len, 300)

        # print(des_sampled_embs.shape)
        # print(des_baseline_embs.shape)

        # load into listener data
        # NOTE now the current batch size is sample_topn times the original batch size
        data_dict["sampled_topn"] = sample_topn
        data_dict["lang_feat"] = {
            "sampled": des_sampled_embs,
            "baseline": des_baseline_embs
        }
        data_dict["lang_len"] = {
            "sampled": des_sampled_lens,
            "baseline": des_baseline_lens
        }


        # fake GT generated by the speaker
        assigned_bbox_id_labels = data_dict["assigned_bbox_id_labels"] # batch_size

        # ref_box_corner_label should be in shape (batch_size, chunk_size, 8, 3)
        # ref_cat_label should be in shape (batch_size, chunk_size)
        assigned_bbox_id_labels = assigned_bbox_id_labels.reshape(-1, chunk_size)
        assigned_bbox_id_labels = assigned_bbox_id_labels.unsqueeze(1).repeat(1, sample_topn, 1)
        assigned_bbox_id_labels = assigned_bbox_id_labels.reshape(-1, chunk_size)

        bbox_corners = data_dict["proposal_bbox_batched"] # batch_size, num_proposals, 8, 3
        bbox_sems = data_dict["proposal_sem_cls_batched"] # batch_size, num_proposals
        _, num_proposals, _, _ = bbox_corners.shape

        # print(assigned_bbox_id_labels.shape)
        # print(bbox_corners.shape)
        # print(bbox_sems.shape)

        # bboxes from detector must be expanded to match the description assigments
        # NOTE only need to duplicate to the sample_topn dim
        bbox_corners = bbox_corners.reshape(-1, 1, 1, num_proposals, 8, 3).repeat(1, sample_topn, chunk_size, 1, 1, 1)
        bbox_corners = bbox_corners.reshape(-1, chunk_size, num_proposals, 8, 3)
        bbox_sems = bbox_sems.reshape(-1, 1, 1, num_proposals).repeat(1, sample_topn, chunk_size, 1)
        bbox_sems = bbox_sems.reshape(-1, chunk_size, num_proposals)

        assert assigned_bbox_id_labels.shape[:2] == bbox_corners.shape[:2]

        # print(bbox_corners.shape)
        # print(bbox_sems.shape)
        # print(assigned_bbox_id_labels.shape)

        ref_box_corner_label = torch.gather(bbox_corners, 2, assigned_bbox_id_labels.reshape(-1, chunk_size, 1, 1, 1).repeat(1, 1, 1, 8, 3)).squeeze(2) # batch_size, chunk_size, 8, 3
        ref_cat_label = torch.gather(bbox_sems, 2, assigned_bbox_id_labels.reshape(-1, chunk_size, 1)).squeeze(2)# batch_size

        # remap semantic labels
        # NOTE wall and floor are excluded in ScanRefer
        ref_cat_label -= 2
        ref_cat_label[ref_cat_label < 0] = 17

        data_dict["ref_box_corner_label"] = ref_box_corner_label
        data_dict["ref_cat_label"] = ref_cat_label
            
        return data_dict

    # NOTE direct access only during inference
    def forward(self, data_dict):

        if not self.no_detection:
            #######################################
            #                                     #
            #           DETECTION BRANCH          #
            #                                     #
            #######################################

            data_dict = self.detector.feed(data_dict, self.current_epoch)

        if not self.no_captioning:
            #######################################
            #                                     #
            #           DETECTION BRANCH          #
            #                                     #
            #######################################

            data_dict = self.speaker(data_dict)

        if not self.no_grounding:
            ########################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################

            # --------- PROPOSAL MATCHING ---------
            data_dict = self.listener(data_dict)

        return data_dict
