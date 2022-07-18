import os
import sys
import torch
import random

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.utils.nn_distance import nn_distance
from lib.utils.bbox import get_aabb3d_iou, get_aabb3d_iou_batch

class TopDownSceneCaptionModule(nn.Module):
    def __init__(self, cfg, vocabulary, embeddings, emb_size=300, feat_size=128, hidden_size=512, 
        num_proposals=256, num_locals=-1, query_mode="corner",
        use_relation=False, use_oracle=False):
        super().__init__() 

        self.cfg = cfg
        self.vocabulary = vocabulary
        # self.embeddings = embeddings
        self.num_vocabs = len(vocabulary["word2idx"])

        self.register_buffer("embeddings", torch.FloatTensor(embeddings))

        self.emb_size = emb_size
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.num_proposals = num_proposals
        self.num_locals = num_locals

        self.query_mode = query_mode

        self.use_relation = use_relation
        # if self.use_relation: self.map_rel = nn.Linear(feat_size * 2, feat_size)

        self.use_oracle = use_oracle

        # top-down recurrent module
        self.map_topdown = nn.Linear(hidden_size + feat_size + emb_size, emb_size)
        self.recurrent_cell_1 = nn.GRUCell(
            input_size=emb_size,
            hidden_size=hidden_size
        )

        # top-down attention module
        self.map_feat = nn.Linear(feat_size, hidden_size, bias=False)
        self.map_hidd = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attend = nn.Linear(hidden_size, 1, bias=False)

        # language recurrent module
        self.map_lang = nn.Linear(feat_size + hidden_size, emb_size)
        self.recurrent_cell_2 = nn.GRUCell(
            input_size=emb_size,
            hidden_size=hidden_size
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_vocabs),
        )

    def forward(self, data_dict, use_tf=True, use_rl=False, is_eval=False, beam_opt={}):
        if not is_eval:
            data_dict = self._forward_sample_batch(data_dict, use_tf, use_rl, beam_opt=beam_opt)
        else:
            data_dict = self._forward_scene_batch(data_dict, beam_opt)

        return data_dict

    def step(self, step_word_idx, hiddens, target_feat, obj_feats, object_masks):
        '''
            recurrent step

            Args:
                step_input: current word embedding, (batch_size, emb_size)
                target_feat: object feature of the target object, (batch_size, feat_size)
                obj_feats: object features of all detected objects, (batch_size, num_proposals, feat_size)
                hidden_1: hidden state of top-down recurrent unit, (batch_size, hidden_size)
                hidden_2: hidden state of language recurrent unit, (batch_size, hidden_size)

            Returns:
                hidden_1: hidden state of top-down recurrent unit, (batch_size, hidden_size)
                hidden_2: hidden state of language recurrent unit, (batch_size, hidden_size)
                masks: attention masks on proposals, (batch_size, num_proposals, 1)
        '''

        # split hidden states
        hidden_1, hidden_2 = hiddens[0], hiddens[1]
        batch_size, _ = hidden_1.shape

        # expand

        # embed input word
        step_word_onehot = torch.zeros(batch_size, len(self.vocabulary["word2idx"])).type_as(hidden_1)
        step_word_onehot.scatter_(1, step_word_idx.unsqueeze(1), 1)
        step_input = torch.matmul(step_word_onehot, self.embeddings) # batch_size, emb_size

        # fuse inputs for top-down module
        step_input = torch.cat([step_input, hidden_2, target_feat], dim=-1)
        step_input = self.map_topdown(step_input)

        # top-down recurrence
        hidden_1 = self.recurrent_cell_1(step_input, hidden_1)

        # top-down attention
        combined = self.map_feat(obj_feats) # batch_size, num_proposals, hidden_size
        combined += self.map_hidd(hidden_1).unsqueeze(1) # batch_size, num_proposals, hidden_size
        combined = torch.tanh(combined)
        scores = self.attend(combined) # batch_size, num_proposals, 1
        scores.masked_fill_(object_masks == 0, 0)

        masks = F.softmax(scores, dim=1) # batch_size, num_proposals, 1
        attended = obj_feats * masks
        attended = attended.sum(1) # batch_size, feat_size

        # fuse inputs for language module
        lang_input = torch.cat([attended, hidden_1], dim=-1)
        lang_input = self.map_lang(lang_input)

        # language recurrence
        hidden_2 = self.recurrent_cell_2(lang_input, hidden_2) # num_proposals, hidden_size

        # language classification
        step_output = self.classifier(hidden_2) # batch_size, num_vocabs
        # logprobs = F.log_softmax(step_output, dim=-1)
        logprobs = step_output.clone()

        # stack hidden states
        hiddens = (hidden_1, hidden_2)

        return step_output, logprobs, hiddens, masks

    # modified from https://github.com/ruotianluo/self-critical.pytorch/blob/master/captioning/models/CaptionModel.py#L35
    def beam_search(self, init_state, init_logprobs, seq_length, target_feat, obj_feats, masks, *args, **kwargs):

        # function computes the similarity score to be augmented
        def add_diversity(beam_seq_table, logprobs, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobs = logprobs.clone()
            batch_size = beam_seq_table[0].shape[0]

            if divm > 0:
                change = logprobs.new_zeros(batch_size, logprobs.shape[-1])
                for prev_choice in range(divm):
                    prev_decisions = beam_seq_table[prev_choice][:, :, local_time] # Nxb
                    for prev_labels in range(bdash):
                        change.scatter_add_(1, prev_decisions[:, prev_labels].unsqueeze(-1), change.new_ones(batch_size, 1))
                
                if local_time == 0:
                    logprobs = logprobs - change * diversity_lambda
                else:
                    logprobs = logprobs - self.repeat_tensor(bdash, change) * diversity_lambda 

            return logprobs, unaug_logprobs


        # does one step of classical beam search

        def beam_step(logprobs, unaug_logprobs, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            #INPUTS:
            #logprobs: probabilities augmented after diversity N*bxV
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions Nxbxl
            #beam_seq_logprobs : log-probability of each decision made, NxbxlxV
            #beam_logprobs_sum : joint log-probability of each beam Nxb

            batch_size = beam_logprobs_sum.shape[0]
            vocab_size = logprobs.shape[-1]
            logprobs = logprobs.reshape(batch_size, -1, vocab_size) # NxbxV
            if t == 0:
                assert logprobs.shape[1] == 1
                beam_logprobs_sum = beam_logprobs_sum[:, :1]
            candidate_logprobs = beam_logprobs_sum.unsqueeze(-1) + logprobs # beam_logprobs_sum Nxb logprobs is NxbxV
            ys, ix = torch.sort(candidate_logprobs.reshape(candidate_logprobs.shape[0], -1), -1, True)
            ys, ix = ys[:,:beam_size], ix[:,:beam_size]
            beam_ix = ix // vocab_size # Nxb which beam
            selected_ix = ix % vocab_size # Nxb # which world
            state_ix = (beam_ix + torch.arange(batch_size).type_as(beam_ix).unsqueeze(-1) * logprobs.shape[1]).reshape(-1) # N*b which in Nxb beams


            if t > 0:
                # gather according to beam_ix
                assert (beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq)) == beam_seq.reshape(-1, beam_seq.shape[-1])[state_ix].view_as(beam_seq)).all()
                beam_seq = beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq))
                
                beam_seq_logprobs = beam_seq_logprobs.gather(1, beam_ix.unsqueeze(-1).unsqueeze(-1).expand_as(beam_seq_logprobs))
            
            beam_seq = torch.cat([beam_seq, selected_ix.unsqueeze(-1)], -1) # beam_seq Nxbxl
            beam_logprobs_sum = beam_logprobs_sum.gather(1, beam_ix) + \
                logprobs.reshape(batch_size, -1).gather(1, ix)
            assert (beam_logprobs_sum == ys).all()
            _tmp_beam_logprobs = unaug_logprobs[state_ix].reshape(batch_size, -1, vocab_size)
            beam_logprobs = unaug_logprobs.reshape(batch_size, -1, vocab_size).gather(1, beam_ix.unsqueeze(-1).expand(-1, -1, vocab_size)) # NxbxV
            assert (_tmp_beam_logprobs == beam_logprobs).all()
            beam_seq_logprobs = torch.cat([
                beam_seq_logprobs,
                beam_logprobs.reshape(batch_size, -1, 1, vocab_size)], 2)
            
            new_state = [None for _ in state]
            for _ix in range(len(new_state)):
            #  copy over state in previous beam q to new beam at vix
                # new_state[_ix] = state[_ix][:, state_ix]
                new_state[_ix] = state[_ix][state_ix]

            state = new_state
            return beam_seq,beam_seq_logprobs,beam_logprobs_sum,state

        # Start diverse_beam_search
        opt = kwargs['opt']
        temperature = opt.get('temperature', 1) # This should not affect beam search, but will affect dbs
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        suppress_UNK = opt.get('suppress_UNK', 0)
        # length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))
        bdash = beam_size // group_size # beam per group

        batch_size = init_logprobs.shape[0]
        device = init_logprobs.device
        # INITIALIZATIONS
        beam_seq_table = [torch.LongTensor(batch_size, bdash, 0).to(device) for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(batch_size, bdash, 0, len(self.vocabulary["word2idx"])).to(device) for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(batch_size, bdash).to(device) for _ in range(group_size)]

        # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
        done_beams_table = [[[] for __ in range(group_size)] for _ in range(batch_size)]
        # state_table = [list(torch.unbind(_)) for _ in torch.stack(init_state).chunk(group_size, 2)]
        # state_table = list(zip(*[_.reshape(-1, batch_size * bdash, group_size, *_.shape[2:]).chunk(group_size, 2) for _ in init_state]))
        state_table = [[_.clone() for _ in init_state] for _ in range(group_size)]
        # logprobs_table = list(init_logprobs.reshape(batch_size * bdash, group_size, -1).chunk(group_size, 0))
        logprobs_table = [init_logprobs.clone() for _ in range(group_size)]
        # END INIT

        # # Chunk elements in the args
        # args = list(args)
        # args = utils.split_tensors(group_size, args) # For each arg, turn (Bbg)x... to (Bb)x(g)x...
        # if self.__class__.__name__ == 'AttEnsemble':
        #     args = [[[args[j][i][k] for i in range(len(self.models))] for j in range(len(args))] for k in range(group_size)] # group_name, arg_name, model_name
        # else:
        #     args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        for t in range(seq_length + group_size - 1):
            for divm in range(group_size): 
                if t >= divm and t <= seq_length + divm - 1:
                    # add diversity
                    logprobs = logprobs_table[divm]

                    # suppress previous word
                    if decoding_constraint and t-divm > 0:
                        logprobs.scatter_(1, beam_seq_table[divm][:, :, t-divm-1].reshape(-1, 1).to(device), float('-inf'))
                    if remove_bad_endings and t-divm > 0:
                        logprobs[torch.from_numpy(np.isin(beam_seq_table[divm][:, :, t-divm-1].cpu().numpy(), self.bad_endings_ix)).reshape(-1), 0] = float('-inf')
                    # suppress UNK tokens in the decoding
                    if suppress_UNK and hasattr(self, 'vocabulary') and self.vocabulary[str(logprobs.size(1)-1)] == 'unk':
                        logprobs[:,logprobs.size(1)-1] = logprobs[:, logprobs.size(1)-1] - 1000  
                    # diversity is added here
                    # the function directly modifies the logprobs values and hence, we need to return
                    # the unaugmented ones for sorting the candidates in the end. # for historical
                    # reasons :-)
                    logprobs, unaug_logprobs = add_diversity(beam_seq_table,logprobs,t,divm,diversity_lambda,bdash)

                    # infer new beams
                    beam_seq_table[divm],\
                    beam_seq_logprobs_table[divm],\
                    beam_logprobs_sum_table[divm],\
                    state_table[divm] = beam_step(logprobs,
                                                unaug_logprobs,
                                                bdash,
                                                t-divm,
                                                beam_seq_table[divm],
                                                beam_seq_logprobs_table[divm],
                                                beam_logprobs_sum_table[divm],
                                                state_table[divm])

                    # if time's up... or if end token is reached then copy beams
                    for b in range(batch_size):
                        is_end = beam_seq_table[divm][b, :, t-divm] == self.vocabulary["word2idx"]["eos"]
                        assert beam_seq_table[divm].shape[-1] == t-divm+1
                        if t == seq_length + divm - 1:
                            is_end.fill_(1)
                        for vix in range(bdash):
                            if is_end[vix]:
                                final_beam = {
                                    'seq': beam_seq_table[divm][b, vix].clone(), 
                                    'logps': beam_seq_logprobs_table[divm][b, vix].clone(),
                                    'unaug_p': beam_seq_logprobs_table[divm][b, vix].sum().item(),
                                    'p': beam_logprobs_sum_table[divm][b, vix].item()
                                }
                                # final_beam['p'] = length_penalty(t-divm+1, final_beam['p'])
                                done_beams_table[b][divm].append(final_beam)
                        beam_logprobs_sum_table[divm][b, is_end] -= 1000

                    # move the current group one step forward in time
                    it = beam_seq_table[divm][:, :, t-divm].reshape(-1).to(logprobs.device)
                    
                    inter_ids = torch.arange(batch_size).repeat_interleave(beam_size)
                    _, logprobs_table[divm], state_table[divm], _ = self.step(
                        it, state_table[divm], target_feat[inter_ids], obj_feats[inter_ids], masks[inter_ids])
                    logprobs_table[divm] = F.log_softmax(logprobs_table[divm] / temperature, dim=-1)

        # all beams are sorted by their log-probabilities
        done_beams_table = [[sorted(done_beams_table[b][i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)] for b in range(batch_size)]
        done_beams = [sum(_, []) for _ in done_beams_table]
        
        return done_beams

    """batched beam decoding - returns the sampled word id and logprobs"""
    def beam_decode(self, target_feats, obj_feats, valid_masks,
        beam_size, max_len):

        batch_size = target_feats.shape[0]

        # start
        start_word_idx = self.vocabulary["word2idx"]["sos"]
        start_word_idx = torch.Tensor([start_word_idx]).type_as(obj_feats).long().repeat(batch_size) # batch_size

        # init hiddens
        hidden_1 = torch.zeros(batch_size, self.hidden_size).type_as(obj_feats) # batch_size, hidden_size
        hidden_2 = torch.zeros(batch_size, self.hidden_size).type_as(obj_feats) # batch_size, hidden_size
        hiddens = (hidden_1, hidden_2)

        _, start_logprobs, hiddens, _ = self.step(
            start_word_idx, 
            hiddens, 
            target_feats, obj_feats, 
            valid_masks)
        start_logprobs = F.log_softmax(start_logprobs, dim=-1)

        # beam search
        done_beams = self.beam_search(
            init_state=hiddens, 
            init_logprobs=start_logprobs, 
            seq_length=max_len, 
            target_feat=target_feats,
            obj_feats=obj_feats,
            masks=valid_masks,
            opt={"beam_size": beam_size}
        )

        return done_beams

    """batched greedy decoding"""
    @torch.no_grad()
    def greedy_decode(self, target_feats, obj_feats, valid_masks, max_len):

        batch_size = target_feats.shape[0]

        step_word_idx = int(self.vocabulary["word2idx"]["sos"])
        step_word_idx = torch.Tensor([step_word_idx]).type_as(target_feats).long() # 1
        step_word_idx = step_word_idx.repeat(batch_size) # batch_size

        # init hiddens
        hidden_1 = torch.zeros(batch_size, self.hidden_size).type_as(target_feats) # batch_size, hidden_size
        hidden_2 = torch.zeros(batch_size, self.hidden_size).type_as(target_feats) # batch_size, hidden_size
        hiddens = (hidden_1, hidden_2)

        outputs = []
        logprobs = []
        for _ in range(max_len):
            _, step_logprobs, hiddens, _ = self.step(step_word_idx, hiddens, target_feats, obj_feats, valid_masks)
            step_logprobs = F.log_softmax(step_logprobs, dim=-1)
            step_logprobs, step_word_idx = step_logprobs.max(-1) # batch_size

            outputs.append(step_word_idx.unsqueeze(1))
            logprobs.append(step_logprobs.unsqueeze(1))

        # aggregate
        outputs = torch.cat(outputs, dim=1).unsqueeze(1) # batch_size, 1, max_len
        logprobs = torch.cat(logprobs, dim=1).unsqueeze(1) # batch_size, 1, max_len

        # trim
        outputs, logprobs = self.trim_outputs(outputs, logprobs)

        return outputs, logprobs

    def trim_outputs(self, raw_word_ids, raw_logprobs):

        batch_size, sample_topn, max_len = raw_word_ids.shape

        # trim the outputs
        trimmed_outputs = []
        trimmed_logprobs = []
        for batch_id in range(batch_size):
            batch_outputs = []
            batch_logprobs = []
            for sample_id in range(sample_topn):
                sample_outputs = raw_word_ids[batch_id, sample_id]
                sample_logprobs = raw_logprobs[batch_id, sample_id]

                # trim all tokens after the first eos or pad_
                masks = torch.ones(max_len)
                for t in range(max_len):
                    cur_token = sample_outputs[t]
                    if (cur_token == int(self.vocabulary["word2idx"]["eos"])) or (cur_token == int(self.vocabulary["word2idx"]["pad_"])):
                        break

                masks[t:] = 0

                batch_outputs.append(sample_outputs[masks == 1])
                batch_logprobs.append(sample_logprobs[masks == 1])

            trimmed_outputs.append(batch_outputs)
            trimmed_logprobs.append(batch_logprobs)

        return trimmed_outputs, trimmed_logprobs

    def select_target(self, bbox_objness, bbox_center, bbox_corner,
        bbox_center_label, bbox_corner_label, ref_box_label, ref_box_corner_label, 
        is_annotated, 
        bbox_id_label=None):
        """
            bbox_objness: (batch_size, num_proposals)
            bbox_center: (batch_size, num_proposals, 3)
            bbox_corner: (batch_size, num_proposals, 8, 3)
            bbox_center_label: (batch_size, max_num_obj, 3)
            bbox_corner_label: (batch_size, max_num_obj, 8, 3)
            ref_box_label: (batch_size, max_num_obj)
            ref_box_corner_label: (batch_size, 8, 3)
            is_annotated: (batch_size,)
        """
        
        # predicted bbox
        batch_size, num_proposals, _ = bbox_center.shape
        
        target_ids = []
        target_ious = []
        assigned_bbox_id_labels = []
        for batch_id in range(batch_size):
            # data in batch
            batch_bbox_objness = bbox_objness[batch_id]
            batch_bbox_center = bbox_center[batch_id]
            batch_bbox_corner = bbox_corner[batch_id]
            batch_bbox_center_label = bbox_center_label[batch_id]
            batch_bbox_corner_label = bbox_corner_label[batch_id]
            batch_ref_box_label = ref_box_label[batch_id]
            batch_ref_box_corner_label = ref_box_corner_label[batch_id]
            batch_is_annotated = is_annotated[batch_id]

            if batch_is_annotated == 1:
                if self.use_oracle:
                    batch_bbox_id_label = bbox_id_label[batch_id]
                    # assigned_bbox_id_label = batch_ref_box_label.argmax(-1).item()
                    assigned_bbox_id_label = bbox_id_label[batch_id]

                    # store
                    target_ids.append(batch_bbox_id_label)
                    target_ious.append(1)
                    assigned_bbox_id_labels.append(assigned_bbox_id_label)
                else:
                    # convert the bbox parameters to bbox corners
                    pred_bbox_batch = batch_bbox_corner # num_proposals, 8, 3
                    gt_bbox_batch = batch_ref_box_corner_label.unsqueeze(0).repeat(num_proposals, 1, 1) # num_proposals, 8, 3
                    ious = get_aabb3d_iou_batch(pred_bbox_batch.detach().cpu().numpy(), gt_bbox_batch.detach().cpu().numpy())
                    target_id = ious.argmax().item() # 0 ~ num_proposals - 1
                    target_iou = ious[target_id].item()
                    assigned_bbox_id_label = batch_ref_box_label.argmax(-1).item()

                    # store
                    target_ids.append(target_id)
                    target_ious.append(target_iou)
                    assigned_bbox_id_labels.append(assigned_bbox_id_label)
            else:
                if self.use_oracle:
                    target_mask = batch_bbox_objness == 1

                    ids = torch.arange(num_proposals).type_as(bbox_corner).long()
                    valid_ids = ids[target_mask == 1] # randomly select object with more than min_num_pts points
                    assert len(valid_ids) > 0
                    target_id = random.choice(valid_ids)

                    batch_bbox_id_label = target_id
                    assigned_bbox_id_label = target_id

                    # store
                    target_ids.append(batch_bbox_id_label)
                    target_ious.append(1)
                    assigned_bbox_id_labels.append(assigned_bbox_id_label)
                else:
                    ids = torch.arange(num_proposals).type_as(bbox_corner).long()
                    valid_ids = ids[batch_bbox_objness == 1] # random select in non-empty predictions
                    target_id = random.choice(valid_ids) if len(valid_ids) > 0 else random.choice(ids)

                    _, object_assignments, _, _ = nn_distance(batch_bbox_center.unsqueeze(0), batch_bbox_center_label.unsqueeze(0))
                    assigned_bbox_id_label = object_assignments[0, target_id]
                    
                    target_bbox = batch_bbox_corner[target_id] # 8, 3
                    batch_ref_box_corner_label = batch_bbox_corner_label[assigned_bbox_id_label] # 8, 3
                    target_iou = get_aabb3d_iou(target_bbox.detach().cpu().numpy(), batch_ref_box_corner_label.detach().cpu().numpy())

                    # store
                    target_ids.append(target_id.item())
                    target_ious.append(target_iou.item())
                    assigned_bbox_id_labels.append(assigned_bbox_id_label.item())

        target_ids = torch.LongTensor(target_ids).type_as(bbox_center).long() # batch_size
        target_ious = torch.FloatTensor(target_ious).type_as(bbox_center) # batch_size
        assigned_bbox_id_labels = torch.LongTensor(assigned_bbox_id_labels).type_as(bbox_center).long() # batch_size

        return target_ids, target_ious, assigned_bbox_id_labels

    def _forward_sample_batch(self, data_dict, use_tf, use_rl, beam_opt={}):
        """
            generate descriptions based on input tokens and object features
        """

        # unpack
        # language-related
        is_annotated = data_dict["annotated"] # batch_size, chunk_size
        # word_embs = data_dict["lang_feat"] # batch_size, max_len, emb_size
        word_ids = data_dict["lang_ids"] # batch_size, chunk_size, max_len
        des_lens = data_dict["lang_len"] # batch_size, chunk_size
        ref_obj_labels = data_dict["ref_box_label"] # batch_size, num_max_obj
        ref_obj_corner_labels = data_dict["ref_box_corner_label"] # batch_size, 8, 3
        # detection-related
        obj_center_labels = data_dict["center_label"] # batch_size, num_max_obj, 3
        obj_corner_labels = data_dict["gt_bbox"] # batch_size, num_max_obj, 8, 3
        obj_feats = data_dict["bbox_feature"] # batch_size, num_proposals, feat_size
        obj_centers = data_dict["proposal_center_batched"] # batch_size, num_proposals, 3
        obj_corners = data_dict["proposal_bbox_batched"] # batch_size, num_proposals, 8, 3
        obj_masks = data_dict["proposal_batch_mask"] # batch_size, num_proposals 
        rel_feats = data_dict["edge_feature"] # batch_size, num_proposals, num_locals, feat_size
        adj_mat = data_dict["adjacent_mat"] # batch_size, num_proposals, num_proposals
        bbox_id_labels = data_dict["bbox_idx"] if self.use_oracle else None
        

        # NOTE chunked language data must be converted to batches
        is_annotated = is_annotated.reshape(-1)
        word_ids = word_ids.reshape(-1, self.cfg.data.max_spk_len+2)
        des_lens = des_lens.reshape(-1)
        ref_obj_labels = ref_obj_labels.reshape(-1, 128)
        ref_obj_corner_labels = ref_obj_corner_labels.reshape(-1, 8, 3)

        num_words = des_lens.max()
        batch_size = des_lens.shape[0] # NOTE batch_size is essentially real batch size times chunk size
        chunk_size = batch_size // obj_center_labels.shape[0]

        # expand detection data
        obj_center_labels = obj_center_labels.unsqueeze(1).repeat(1, chunk_size, 1, 1).reshape(batch_size, 128, 3)
        obj_corner_labels = obj_corner_labels.unsqueeze(1).repeat(1, chunk_size, 1, 1, 1).reshape(batch_size, 128, 8, 3)
        obj_feats = obj_feats.unsqueeze(1).repeat(1, chunk_size, 1, 1).reshape(batch_size, self.num_proposals, -1)
        obj_centers = obj_centers.unsqueeze(1).repeat(1, chunk_size, 1, 1).reshape(batch_size, self.num_proposals, 3)
        obj_corners = obj_corners.unsqueeze(1).repeat(1, chunk_size, 1, 1, 1).reshape(batch_size, self.num_proposals, 8, 3)
        obj_masks = obj_masks.unsqueeze(1).repeat(1, chunk_size, 1).reshape(batch_size, self.num_proposals)
        rel_feats = rel_feats.unsqueeze(1).repeat(1, chunk_size, 1, 1, 1).reshape(batch_size, self.num_proposals, self.num_locals, -1)
        adj_mat = adj_mat.unsqueeze(1).repeat(1, chunk_size, 1, 1).reshape(batch_size, self.num_proposals, self.num_proposals)
        bbox_id_labels = bbox_id_labels.reshape(-1) if self.use_oracle else None


        # find the target object ids
        # is_annotated = des_lens > 0 # annotated if description length is not 0
        target_ids, target_ious, assigned_bbox_id_labels = self.select_target(
            bbox_objness=obj_masks,
            bbox_center=obj_centers,
            bbox_corner=obj_corners,
            bbox_center_label=obj_center_labels,
            bbox_corner_label=obj_corner_labels,
            ref_box_label=ref_obj_labels,
            ref_box_corner_label=ref_obj_corner_labels, 
            is_annotated=is_annotated,
            bbox_id_label=bbox_id_labels if self.use_oracle else None
        )

        # store
        data_dict["assigned_bbox_id_labels"] = assigned_bbox_id_labels

        # select object features
        target_feats = torch.gather(
            obj_feats, 1, target_ids.view(batch_size, 1, 1).repeat(1, 1, self.feat_size)).squeeze(1) # batch_size, emb_size

        # valid object proposal masks
        valid_masks = obj_masks if self.num_locals == -1 else self._query_locals(obj_corners, target_ids, obj_masks)
        valid_masks = valid_masks.unsqueeze(-1)

        # object-to-object relation
        if self.use_relation:
            obj_feats = self._add_relation_feat(rel_feats, adj_mat, obj_feats, target_ids)

        # recurrent from 0 to max_len - 2
        if use_rl:
            assert beam_opt

            beam_size = beam_opt.get("train_beam_size", 5)
            sample_topn = beam_opt.get("train_sample_topn", 1)

            outputs = [] # output captions
            logprobs = [] # output logprobs
            baseline_outputs = [] # baseline method captions
            assigned_bbox_id_labels = []
            
            # ----- beam decoding -------
            done_beams = self.beam_decode(
                target_feats=target_feats, 
                obj_feats=obj_feats, 
                valid_masks=valid_masks,
                beam_size=beam_size, 
                max_len=self.cfg.data.max_spk_len
            )

            # aggregate outputs
            for batch_id in range(batch_size):
                batch_outputs = []
                batch_logprobs = []
                for beam_id in range(sample_topn):
                    seq = done_beams[batch_id][beam_id]["seq"]
                    logps = done_beams[batch_id][beam_id]["logps"].gather(1, seq.unsqueeze(1)).squeeze(1)
                    batch_outputs.append(seq)
                    batch_logprobs.append(logps)

                outputs.append(batch_outputs)
                logprobs.append(batch_logprobs)

            # ----- greedy decoding -------
            baseline_outputs, _ = self.greedy_decode(
                target_feats=target_feats, 
                obj_feats=obj_feats, 
                valid_masks=valid_masks,
                max_len=self.cfg.data.max_spk_len+1
            )

            # expand
            baseline_outputs = [[baseline_outputs[i][0] for _ in range(sample_topn)] for i in range(batch_size)]

            # store
            data_dict["lang_logprob"] = logprobs
            data_dict["baseline_cap"] = baseline_outputs

        else:
            outputs = []
            masks = []
            hidden_1 = torch.zeros(batch_size, self.hidden_size).type_as(obj_feats) # batch_size, hidden_size
            hidden_2 = torch.zeros(batch_size, self.hidden_size).type_as(obj_feats) # batch_size, hidden_size
            hiddens = (hidden_1, hidden_2)
            step_id = 0
            # step_input = word_embs[:, step_id] # batch_size, emb_size
            step_word_idx = word_ids[:, step_id] # batch_size
            while True:
                # feed
                step_output, _, hiddens, step_mask = self.step(
                    step_word_idx, hiddens, target_feats, obj_feats, valid_masks)
                
                # # differentiable sampling
                # step_pred_onehot = F.gumbel_softmax(step_output, hard=True, tau=0.1) # batch_size, num_vocabs

                # non-differentiable samping
                step_output_ids = step_output.argmax(-1) # batch_size

                # store
                step_output = step_output.unsqueeze(1) # batch_size, 1, num_vocabs 
                outputs.append(step_output)
                masks.append(step_mask) # batch_size, num_proposals, 1

                # next step
                step_id += 1
                if step_id == num_words - 1: break # exit for train mode
                step_word_idx = word_ids[:, step_id] if use_tf else step_output_ids # batch_size, emb_size

            outputs = torch.cat(outputs, dim=1) # batch_size, num_words - 1/max_len, num_vocabs
            masks = torch.cat(masks, dim=-1) # batch_size, num_proposals, num_words - 1/max_len
        
            # store
            data_dict["topdown_attn"] = masks


        # NOTE when the IoU of best matching predicted boxes and the GT boxes 
        # are smaller than the threshold, the corresponding predicted captions
        # should be filtered out in case the model learns wrong things
        good_bbox_masks = target_ious > self.cfg.data.min_iou_threshold # batch_size

        num_good_bboxes = good_bbox_masks.sum()
        mean_target_ious = target_ious[good_bbox_masks].mean() if num_good_bboxes > 0 else torch.zeros(1)[0].type_as(obj_feats)

        # store
        data_dict["lang_cap"] = outputs
        data_dict["pred_ious"] = mean_target_ious
        data_dict["valid_masks"] = valid_masks
        data_dict["good_bbox_masks"] = good_bbox_masks

        return data_dict

    def _forward_scene_batch(self, data_dict, beam_opt={}):
        """
        generate descriptions based on input tokens and object features
        """

        # unpack
        word_ids = data_dict["lang_ids"] # batch_size, chunk_size, max_len
        word_embs = data_dict["lang_feat"] # batch_size, emb_size
        obj_feats = data_dict["bbox_feature"] # batch_size, num_proposals, feat_size
        obj_masks = data_dict["proposal_batch_mask"] # batch_size, num_proposals
        obj_corners = data_dict["proposal_bbox_batched"] # batch_size, num_proposals, 8, 3
        rel_feats = data_dict["edge_feature"] # batch_size, num_proposals, num_locals, feat_size
        adj_mat = data_dict["adjacent_mat"] # batch_size, num_proposals, num_proposals
        batch_size = word_embs.shape[0]

        # beam_size = beam_opt.get("eval_beam_size", self.cfg.train.beam_size)

        # recurrent from 0 to max_len - 2
        outputs = []
        masks = []
        valid_masks = []
        for prop_id in range(self.num_proposals):
            # select object features
            target_feats = obj_feats[:, prop_id] # batch_size, emb_size
            target_ids = torch.zeros(batch_size).fill_(prop_id).type_as(target_feats).long()

            prop_obj_feats = obj_feats.clone()
            # valid_prop_masks = self._get_valid_object_masks(data_dict, target_ids, object_masks)
            valid_prop_masks = obj_masks if self.num_locals == -1 else self._query_locals(obj_corners, target_ids, obj_masks)

            # object-to-object relation
            if self.use_relation:
                prop_obj_feats = self._add_relation_feat(rel_feats, adj_mat, prop_obj_feats, target_ids)
                # valid_prop_masks = self._expand_object_mask(data_dict, valid_prop_masks, self.num_locals)

            valid_masks.append(valid_prop_masks.unsqueeze(1))

            # start recurrence
            prop_outputs = []
            prop_masks = []
            hidden_1 = torch.zeros(batch_size, self.hidden_size).type_as(target_feats) # batch_size, hidden_size
            hidden_2 = torch.zeros(batch_size, self.hidden_size).type_as(target_feats) # batch_size, hidden_size
            hiddens = (hidden_1, hidden_2)
            step_id = 0
            step_input = self.vocabulary["word2idx"]["sos"]
            step_input = torch.Tensor([step_input]).repeat(batch_size).type_as(obj_feats).long() # batch_size
            while True:
                # feed
                step_output, _, hiddens, step_mask = self.step(
                    step_input, hiddens, target_feats, obj_feats, valid_prop_masks.unsqueeze(-1))
                
                step_output = step_output.argmax(-1)

                # store
                prop_outputs.append(step_output.unsqueeze(1))
                prop_masks.append(step_mask)

                # next step
                step_id += 1
                if step_id == self.cfg.data.max_spk_len + 1: break # exit for eval mode
                step_input = step_output # batch_size

                # # teacher forcing
                # # NOTE only enabled for debugging
                # gt_word = word_ids[:, 0, step_id]
                # step_input = gt_word

            prop_outputs = torch.cat(prop_outputs, dim=1).unsqueeze(1) # batch_size, 1, num_words - 1/max_len
            prop_masks = torch.cat(prop_masks, dim=-1).unsqueeze(1) # batch_size, 1, num_proposals, num_words - 1/max_len
            outputs.append(prop_outputs)
            masks.append(prop_masks)

        outputs = torch.cat(outputs, dim=1) # batch_size, num_proposals, num_words - 1/max_len
        masks = torch.cat(masks, dim=1) # batch_size, num_proposals, num_proposals, num_words - 1/max_len
        valid_masks = torch.cat(valid_masks, dim=1) # batch_size, num_proposals, num_proposals

        # store
        data_dict["lang_cap"] = outputs
        data_dict["topdown_attn"] = masks
        data_dict["valid_masks"] = valid_masks

        return data_dict

    def _nn_distance(self, pc1, pc2):
        """
        Input:
            pc1: (B,N,C) torch tensor
            pc2: (B,M,C) torch tensor

        Output:
            dist1: (B,N) torch float32 tensor
            idx1: (B,N) torch int64 tensor
            dist2: (B,M) torch float32 tensor
            idx2: (B,M) torch int64 tensor
        """

        N = pc1.shape[1]
        M = pc2.shape[1]
        pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
        pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
        pc_diff = pc1_expand_tile - pc2_expand_tile
        pc_dist = torch.sqrt(torch.sum(pc_diff**2, dim=-1) + 1e-8) # (B,N,M)

        return pc_dist
    
    def _get_bbox_centers(self, corners):
        coord_min = torch.min(corners, dim=2)[0] # batch_size, num_proposals, 3
        coord_max = torch.max(corners, dim=2)[0] # batch_size, num_proposals, 3

        return (coord_min + coord_max) / 2

    def _query_locals(self, corners, target_ids, object_masks, include_self=True, overlay_threshold=0.5):
        centers = self._get_bbox_centers(corners) # batch_size, num_proposals, 3
        batch_size = target_ids.shape[0]

        # decode target box info
        target_centers = torch.gather(centers, 1, target_ids.view(-1, 1, 1).repeat(1, 1, 3)) # batch_size, 1, 3
        target_corners = torch.gather(corners, 1, target_ids.view(-1, 1, 1, 1).repeat(1, 1, 8, 3)) # batch_size, 1, 8, 3

        # get the distance
        if self.query_mode == "center":
            pc_dist = self._nn_distance(target_centers, centers).squeeze(1) # batch_size, num_proposals
        elif self.query_mode == "corner":
            pc_dist = self._nn_distance(target_corners.squeeze(1), centers) # batch_size, 8, num_proposals
            pc_dist, _ = torch.min(pc_dist, dim=1) # batch_size, num_proposals
        else:
            raise ValueError("invalid distance mode, choice: [\"center\", \"corner\"]")

        # mask out invalid objects
        pc_dist.masked_fill_(object_masks == 0, float('1e30')) # distance to invalid objects: infinity

        # exclude overlaid boxes
        tar2neigbor_iou = get_aabb3d_iou_batch(
            target_corners.repeat(1, self.num_proposals, 1, 1).view(-1, 8, 3).detach().cpu().numpy(), 
            corners.view(-1, 8, 3).detach().cpu().numpy()
        )
        tar2neigbor_iou = torch.from_numpy(tar2neigbor_iou).type_as(pc_dist).view(batch_size, self.num_proposals) # batch_size, num_proposals
        overlaid_masks = tar2neigbor_iou >= overlay_threshold
        pc_dist.masked_fill_(overlaid_masks, float('1e30')) # distance to overlaid objects: infinity

        # include the target objects themselves
        self_dist = 0 if include_self else float('1e30')
        self_masks = torch.zeros(batch_size, self.num_proposals).type_as(corners)
        self_masks.scatter_(1, target_ids.view(-1, 1), 1)
        pc_dist.masked_fill_(self_masks == 1, self_dist) # distance to themselves: 0 or infinity

        # get the top-k object ids
        _, topk_ids = torch.topk(pc_dist, self.num_locals, largest=False, dim=1) # batch_size, num_locals

        # construct masks for the local context
        local_masks = torch.zeros(batch_size, self.num_proposals).type_as(corners)
        local_masks.scatter_(1, topk_ids, 1)

        return local_masks

    def _create_adjacent_mat(self, data_dict, object_masks):
        batch_size, num_objects = object_masks.shape
        adjacent_mat = torch.zeros(batch_size, num_objects, num_objects).type_as(object_masks)

        for obj_id in range(num_objects):
            target_ids = torch.LongTensor([obj_id for _ in range(batch_size)]).type_as(object_masks).long()
            adjacent_entry = self._query_locals(data_dict, target_ids, object_masks, include_self=False) # batch_size, num_objects
            adjacent_mat[:, obj_id] = adjacent_entry

        return adjacent_mat

    def _get_valid_object_masks(self, data_dict, target_ids, object_masks):
        if self.num_locals == -1:
            valid_masks = object_masks
        else:
            adjacent_mat = data_dict["adjacent_mat"]
            batch_size, _, _ = adjacent_mat.shape
            valid_masks = torch.gather(
                adjacent_mat, 1, target_ids.view(batch_size, 1, 1).repeat(1, 1, self.num_proposals)).squeeze(1) # batch_size, num_proposals

        return valid_masks

    def _add_relation_feat(self, rel_feats, adjacent_mat, obj_feats, target_ids):
        batch_size = rel_feats.shape[0]

        rel_feats = torch.gather(rel_feats, 1, 
            target_ids.view(batch_size, 1, 1, 1).repeat(1, 1, self.num_locals, self.feat_size)).squeeze(1) # batch_size, num_locals, feat_size

        # new_obj_feats = torch.cat([obj_feats, rel_feats], dim=1) # batch_size, num_proposals + num_locals, feat_size

        # scatter the relation features to objects
        rel_indices = torch.gather(adjacent_mat, 1, 
            target_ids.view(batch_size, 1, 1).repeat(1, 1, self.num_proposals)).squeeze(1) # batch_size, num_proposals
        rel_masks = rel_indices.unsqueeze(-1).repeat(1, 1, self.feat_size) == 1 # batch_size, num_proposals, feat_size
        scattered_rel_feats = torch.zeros(obj_feats.shape).type_as(rel_feats)
        scattered_rel_feats = scattered_rel_feats.masked_scatter(rel_masks, rel_feats) # batch_size, num_proposals, feat_size

        new_obj_feats = obj_feats + scattered_rel_feats
        # new_obj_feats = torch.cat([obj_feats, scattered_rel_feats], dim=-1)
        # new_obj_feats = self.map_rel(new_obj_feats)

        return new_obj_feats

    def _expand_object_mask(self, data_dict, object_masks, num_extra):
        batch_size, num_objects = object_masks.shape
        exp_masks = torch.zeros(batch_size, num_extra).type_as(object_masks)

        num_edge_targets = data_dict["num_edge_target"]
        for batch_id in range(batch_size):
            exp_masks[batch_id, :num_edge_targets[batch_id]] = 1

        object_masks = torch.cat([object_masks, exp_masks], dim=1) # batch_size, num_objects + num_extra

        return object_masks

    