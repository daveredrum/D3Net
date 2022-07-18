# Evaluates semantic label task
# Input:
#   - path to .txt prediction files
#   - path to .txt ground truth files
#

import os
from glob import glob
from importlib import import_module

import numpy as np

from lib.utils.log import Logger
from lib.utils.eval import read_sem_ids


CLASS_NAME = None
CLASS_IDX = None


def build_confusion_for_scene(pred_file, gt_file, confusion):
    pred_ids = read_sem_ids(pred_file, 'pred')
    gt_ids = read_sem_ids(gt_file, 'gt')
    assert pred_ids.size == gt_ids.size
    for i in range(pred_ids.size):
        pred_id = pred_ids[i]
        gt_id = gt_ids[i]
        confusion[gt_id][pred_id] += 1


def get_semantic_iou(class_idx, confusion):
    # if not class_idx in CLASS_IDX:
    #     return float('nan')
    
    # #true positives
    tp = np.longlong(confusion[class_idx, class_idx])
    # #false negatives
    fn = np.longlong(confusion[class_idx, :].sum()) - tp
    # #false positives
    not_ignored = [idx for idx in CLASS_IDX if not idx == class_idx]
    fp = np.longlong(confusion[not_ignored, class_idx].sum())

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return (float(tp) / denom, tp, denom)


def write_result_file(confusion, ious, logger):
    logger.info('iou scores\n')
    for i in range(len(CLASS_IDX)):
        class_idx = CLASS_IDX[i]
        class_name = CLASS_NAME[i]
        iou = ious[class_name][0]
        logger.info(f'{class_name:<14s}({class_idx:<2d}): {iou:>5.3f}\n')
    logger.info('\nconfusion matrix\n')
    logger.info('\t\t\t')
    for i in range(len(CLASS_IDX)):
        logger.info(f'{CLASS_IDX[i]:<8d}')
    logger.info('\n')
    for r in range(len(CLASS_IDX)):
        logger.info(f'{CLASS_NAME[r]:<14s}({CLASS_IDX[r]:<2d})')
        for c in range(len(CLASS_IDX)):
            logger.info(f'\t{confusion[CLASS_IDX[r],CLASS_IDX[c]]:>5.3f}')
        logger.info('\n')
    logger.debug('wrote results to less.log')


def evaluate(pred_files, gt_files, logger):
    max_class_id = np.max(CLASS_IDX) + 1
    confusion = np.zeros((max_class_id, max_class_id), dtype=np.ulonglong)

    logger.debug(f'evaluating {len(pred_files)} scenes...')
    for i in range(len(pred_files)):
        scene_id = pred_files[i].split('/')[-1].split('.')[0]
        build_confusion_for_scene(pred_files[i], gt_files[i], confusion)
        logger.debug(f"{i+1} scenes processed: {scene_id}")

    class_ious = {}
    for i in range(len(CLASS_IDX)):
        class_name = CLASS_NAME[i]
        class_idx = CLASS_IDX[i]
        class_ious[class_name] = get_semantic_iou(class_idx, confusion)

    logger.debug('classes          IoU')
    logger.debug('----------------------------')
    for i in range(len(CLASS_IDX)):
        class_name = CLASS_NAME[i]
        logger.info(f'{class_name:<14s}: {class_ious[class_name][0]:>5.3f}   ({class_ious[class_name][1]:>6d}/{class_ious[class_name][2]:<6d})')
    write_result_file(confusion, class_ious, logger)


def evaluate_semantic(cfg):
    logger = Logger.from_evaluation(cfg)
    
    global CLASS_NAME
    global CLASS_IDX
    # exclude unannotated class label
    CLASS_NAME = getattr(import_module(cfg.evaluation.model_utils_module), cfg.evaluation.gt_class_name)[1:]
    CLASS_IDX = np.array(getattr(import_module(cfg.evaluation.model_utils_module), cfg.evaluation.gt_class_idx))[1:]
    
    pred_path = os.path.join(cfg.OUTPUT_PATH, cfg.general.dataset, cfg.general.model, cfg.evaluation.use_model, "test", cfg.data.split, 'semantic')
    pred_files = sorted(glob(os.path.join(pred_path, '*.txt')))
    
    gt_path = os.path.join(cfg.DATA_PATH, cfg.general.dataset, 'split_gt', cfg.data.split)
    gt_files = sorted(glob(os.path.join(gt_path, '*.txt')))

    # evaluate
    evaluate(pred_files, gt_files, logger)

