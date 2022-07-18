# Evaluates semantic instance task
# Adapted from the CityScapes evaluation: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py
# Input:
#   - path to .txt prediction files
#   - path to .txt ground truth files
# Each .txt prediction file look like:
#    [(pred0) rel. path to pred. mask over verts as .txt] [(pred0) label id] [(pred0) confidence]
#    [(pred1) rel. path to pred. mask over verts as .txt] [(pred1) label id] [(pred1) confidence]
#    [(pred2) rel. path to pred. mask over verts as .txt] [(pred2) label id] [(pred2) confidence]
#    ...
#
# NOTE: The prediction files must live in the root of the given prediction path.
#       Predicted mask .txt files must live in a subfolder.
#       Additionally, filenames must not contain spaces.
# The relative paths to predicted masks must contain one integer per line,
# where each line corresponds to vertices in the *.ply (in that order).
# Non-zero integers indicate part of the predicted instance.
# The label ids specify the class of the corresponding mask.
# Confidence is a float confidence score of the mask.
#
# Note that only the valid classes are used for evaluation,
# i.e., any ground truth label not in the valid label set
# is ignored in the evaluation.


import os
from glob import glob
from importlib import import_module
from copy import deepcopy

import numpy as np

from lib.utils.log import Logger
from lib.utils.eval import read_inst_ids, parse_inst_pred_file, get_instances


# ---------- Label info ---------- #
CLASS_NAME = None
CLASS_IDX = None
IDX_TO_NAME = {}
NAME_TO_IDX = {}

# ---------- Evaluation params ---------- #
# overlaps for evaluation
OVERLAPS             = np.append(np.arange(0.5,0.95,0.05), 0.25)
# minimum region size for evaluation [verts]
MIN_REGION_SIZES     = np.array( [ 100 ] )
# distance thresholds [m]
DISTANCE_THRESHES    = np.array( [  float('inf') ] )
# distance confidences
DISTANCE_CONFS       = np.array( [ -float('inf') ] )


def evaluate_matches(matches):
    overlaps = OVERLAPS
    min_region_sizes = [ MIN_REGION_SIZES[0] ]
    dist_threshes = [ DISTANCE_THRESHES[0] ]
    dist_confs = [ DISTANCE_CONFS[0] ]
    
    # results: class x overlap
    ap = np.zeros( (len(dist_threshes) , len(CLASS_NAME) , len(overlaps)) , np.float )
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]['pred']:
                    for class_name in CLASS_NAME:
                        for pred_inst in matches[m]['pred'][class_name]:
                            if 'filename' in pred_inst:
                                pred_visited[pred_inst['filename']] = False
            for ci, class_name in enumerate(CLASS_NAME):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]['pred'][class_name]
                    gt_instances = matches[m]['gt'][class_name]
                    # filter groups in ground truth
                    gt_instances = [ gt for gt in gt_instances if gt['vert_count']>=min_region_size and gt['med_dist']<=distance_thresh and gt['dist_conf']>=distance_conf ]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true  = np.ones ( len(gt_instances) )
                    cur_score = np.ones ( len(gt_instances) ) * (-float("inf"))
                    cur_match = np.zeros( len(gt_instances) , dtype=np.bool )
                    # collect matches
                    for gti, gt in enumerate(gt_instances):
                        found_match = False
                        num_pred = len(gt['matched_pred'])
                        for pred in gt['matched_pred']:
                            # greedy assignments
                            if pred_visited[pred['filename']]:
                                continue
                            overlap = float(pred['intersection']) / (gt['vert_count']+pred['vert_count']-pred['intersection'])
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max( cur_score[gti] , confidence )
                                    min_score = min( cur_score[gti] , confidence )
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true  = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['filename']] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true  = cur_true [ cur_match==True ]
                    cur_score = cur_score[ cur_match==True ]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred['matched_gt']:
                            overlap = float(gt['intersection']) / (gt['vert_count']+pred['vert_count']-gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                # group?
                                if gt['class_idx'] == 0:
                                    num_ignore += gt['intersection']
                                # small ground truth instances
                                if gt['vert_count']<min_region_size or gt['med_dist']>distance_thresh or gt['dist_conf']<distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(num_ignore)/pred['vert_count']
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true  = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort      = np.argsort(y_score)
                    y_score_sorted      = y_score[score_arg_sort]
                    y_true_sorted       = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    thresholds, unique_indices = np.unique( y_score_sorted , return_index=True )
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples      = len(y_score_sorted)
                    num_true_examples = y_true_sorted_cumsum[-1]
                    precision         = np.zeros(num_prec_recall)
                    recall            = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append( y_true_sorted_cumsum , 0 )
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores-1]
                        tp = num_true_examples - cumsum
                        fp = num_examples      - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p  = float(tp)/(tp+fp)
                        r  = float(tp)/(tp+fn)
                        precision[idx_res] = p
                        recall   [idx_res] = r

                    # first point in curve is artificial
                    precision[-1] = 1.
                    recall   [-1] = 0.

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)

                    stepWidths = np.convolve(recall_for_conv,[-0.5, 0, 0.5],'valid')
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float('nan')
                ap[di,ci,oi] = ap_current
    return ap


def compute_averages(aps):
    d_inf = 0
    o50   = np.where(np.isclose(OVERLAPS, 0.5))
    o25   = np.where(np.isclose(OVERLAPS, 0.25))
    oAllBut25  = np.where(np.logical_not(np.isclose(OVERLAPS, 0.25)))
    avg_dict = {}
    #avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,oAllBut25])
    avg_dict['all_ap_50%'] = np.nanmean(aps[ d_inf,:,o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[ d_inf,:,o25])
    avg_dict["classes"]  = {}
    for (ci, class_name) in enumerate(CLASS_NAME):
        avg_dict["classes"][class_name]             = {}
        #avg_dict["classes"][class_name]["ap"]       = np.average(aps[ d_inf,ci,  :])
        avg_dict["classes"][class_name]["ap"]       = np.average(aps[ d_inf,ci,oAllBut25])
        avg_dict["classes"][class_name]["ap50%"]    = np.average(aps[ d_inf,ci,o50])
        avg_dict["classes"][class_name]["ap25%"]    = np.average(aps[ d_inf,ci,o25])
    return avg_dict


def assign_instances_for_scene(pred_file, gt_file):
    pred_info = parse_inst_pred_file(pred_file)
    gt_sem_idx, gt_inst_ids = read_inst_ids(gt_file, 'gt')

    # get gt instances
    gt_instances = get_instances(gt_sem_idx, gt_inst_ids, CLASS_IDX, CLASS_NAME, IDX_TO_NAME)
    # associate
    gt2pred = deepcopy(gt_instances)
    for class_name in gt2pred:
        for gt_inst in gt2pred[class_name]:
            gt_inst['matched_pred'] = []
    pred2gt = {}
    for class_name in CLASS_NAME:
        pred2gt[class_name] = []
    num_pred_instances = 0
    
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.in1d(gt_sem_idx, CLASS_IDX))
    
    # go through all prediction masks
    for pred_mask_file in pred_info:
        pred_class_idx = int(pred_info[pred_mask_file]['class_idx'])
        confidence = pred_info[pred_mask_file]['confidence']
        if not pred_class_idx in IDX_TO_NAME:
            continue
        pred_class_name = IDX_TO_NAME[pred_class_idx]
        # read the mask
        _, pred_inst_mask = read_inst_ids(pred_mask_file, 'pred')
        assert len(pred_inst_mask) == len(gt_inst_ids)

        # convert to binary
        pred_inst_mask = pred_inst_mask != 0
        num_verts = pred_inst_mask.sum(0)
        if num_verts < MIN_REGION_SIZES[0]:
            continue  # skip if no enough verts

        pred_instance = {}
        pred_instance['filename'] = pred_mask_file
        pred_instance['pred_id'] = num_pred_instances
        pred_instance['class_idx'] = pred_class_idx
        pred_instance['vert_count'] = num_verts
        pred_instance['confidence'] = confidence
        pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_inst_mask))

        # matched gt instances
        matched_gt = []
        # go through all gt instances with matching pred_class_name
        for (gt_num, gt_inst) in enumerate(gt2pred[pred_class_name]):
            intersection = np.count_nonzero(np.logical_and(gt_inst_ids == gt_inst['target_id'], pred_inst_mask))
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy['intersection']   = intersection
                pred_copy['intersection'] = intersection
                matched_gt.append(gt_copy)
                gt2pred[pred_class_name][gt_num]['matched_pred'].append(pred_copy)
        pred_instance['matched_gt'] = matched_gt
        num_pred_instances += 1
        pred2gt[pred_class_name].append(pred_instance)

    return gt2pred, pred2gt


def print_results(avgs, logger):
    sep     = "" 
    col1    = ":"
    lineLen = 64

    logger.info("")
    logger.info("#"*lineLen)
    line  = ""
    line += "{:<15}".format("what"      ) + sep + col1
    line += "{:>15}".format("AP"        ) + sep
    line += "{:>15}".format("AP_50%"    ) + sep
    line += "{:>15}".format("AP_25%"    ) + sep
    logger.info(line)
    logger.info("#"*lineLen)

    for (ci, class_name) in enumerate(CLASS_NAME):
        ap_avg  = avgs["classes"][class_name]["ap"]
        ap_50o  = avgs["classes"][class_name]["ap50%"]
        ap_25o  = avgs["classes"][class_name]["ap25%"]
        line  = "{:<15}".format(class_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_avg ) + sep
        line += sep + "{:>15.3f}".format(ap_50o ) + sep
        line += sep + "{:>15.3f}".format(ap_25o ) + sep
        logger.info(line)

    all_ap_avg  = avgs["all_ap"]
    all_ap_50o  = avgs["all_ap_50%"]
    all_ap_25o  = avgs["all_ap_25%"]

    logger.info("-"*lineLen)
    line  = "{:<15}".format("average") + sep + col1 
    line += "{:>15.3f}".format(all_ap_avg)  + sep 
    line += "{:>15.3f}".format(all_ap_50o)  + sep
    line += "{:>15.3f}".format(all_ap_25o)  + sep
    logger.info(line)
    logger.info("")


def write_result_file(avgs, logger):
    _SPLITTER = ','
    logger.info(_SPLITTER.join(['class', 'class id', 'ap', 'ap50', 'ap25']) + '\n')
    for i in range(len(CLASS_IDX)):
        class_name = CLASS_NAME[i]
        class_id = CLASS_IDX[i]
        ap = avgs["classes"][class_name]["ap"]
        ap50 = avgs["classes"][class_name]["ap50%"]
        ap25 = avgs["classes"][class_name]["ap25%"]
        logger.info(_SPLITTER.join([str(x) for x in [class_name, class_id, ap, ap50, ap25]]) + '\n')    


def evaluate(pred_files, gt_files, logger):
    logger.debug(f'evaluating {len(pred_files)} scenes...')
    matches = {}
    for i in range(len(pred_files)):
        matches_key = os.path.abspath(gt_files[i])
        # assign gt to predictions
        gt2pred, pred2gt = assign_instances_for_scene(pred_files[i], gt_files[i])
        matches[matches_key] = {}
        matches[matches_key]['gt'] = gt2pred
        matches[matches_key]['pred'] = pred2gt
        logger.debug(f"\rscenes processed: {i+1}")

    ap_scores = evaluate_matches(matches)
    avgs = compute_averages(ap_scores)

    # print
    print_results(avgs, logger)
    write_result_file(avgs, logger)


def evaluate_instance(cfg):
    logger = Logger.from_evaluation(cfg)
    
    global CLASS_NAME
    global CLASS_IDX
    global NAME_TO_IDX
    global IDX_TO_NAME
    # CLASS_NAME = getattr(import_module(cfg.evaluation.model_utils_module), cfg.evaluation.gt_class_name)[1:]
    # CLASS_IDX = np.array(getattr(import_module(cfg.evaluation.model_utils_module), cfg.evaluation.gt_class_idx))[1:]
    # for scannet temporarily
    CLASS_NAME = getattr(import_module(cfg.evaluation.model_utils_module), cfg.evaluation.gt_class_name)[3:]
    CLASS_IDX = np.array(getattr(import_module(cfg.evaluation.model_utils_module), cfg.evaluation.gt_class_idx))[3:]
    for i in range(len(CLASS_IDX)):
        NAME_TO_IDX[CLASS_NAME[i]] = CLASS_IDX[i]
        IDX_TO_NAME[CLASS_IDX[i]] = CLASS_NAME[i]
    
    pred_path = os.path.join(cfg.OUTPUT_PATH, cfg.general.dataset, cfg.general.model, cfg.evaluation.use_model, "test", cfg.data.split, "instance")
    pred_files = sorted(glob(os.path.join(pred_path, '*.txt')))
    pred_files = [p for p in pred_files if len(p.split('/')[-1].split('.')) <= 2]
    
    gt_path = os.path.join(cfg.DATA_PATH, cfg.general.dataset, 'split_gt', cfg.data.split)
    gt_files = sorted(glob(os.path.join(gt_path, '*.txt')))

    # evaluate
    evaluate(pred_files, gt_files, logger)