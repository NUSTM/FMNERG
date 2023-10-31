# -*- coding: utf-8 -*-

# This script handles the decoding functions and performance measurement

import re
import torch


def extract_spans_para_quad(seq, vis_label, idx, img_id, object_detection_fault, seq_type, all_pred_sentence_number=0,
                            all_wrong_match_sentence_number=0):
    triplets = {}
    coarse_dict = {}

    sents = [s.strip() for s in seq.split('[SSEP]')]
    for s in sents:
        all_pred_sentence_number += 1
        try:
            part_one, part_two = s.split(', which')
            entity, labels = part_one.split(' is a ')
            coarse_label, fine_label = labels.split(' and a ')
            in_the_image = False if 'not in the image' in part_two else True
            if 'in the image' not in part_two:
                in_the_image = False
            if in_the_image:
                vis = vis_label[idx]
                idx += 1
            else:
                if seq_type == 'gold' and img_id + ' ' + entity in object_detection_fault:
                    in_the_image = True
                vis = [None]

        except:
            try:
                print(f'In {seq_type} seq, cannot decode: {s}, length={len(s)}')
                if len(s) > 0:
                    print("NO MATCHING: " + s)
                    all_wrong_match_sentence_number += 1
                pass
            except UnicodeEncodeError:
                print(f'In {seq_type} seq, a string cannot be decoded')
                pass
            entity, fine_label, coarse_label, in_the_image, vis = '', '', '', '', []
        # triples.append((entity, label, in_the_image, vis))
        triplets[str(entity) + ' ' + str(fine_label) + ' ' + str(in_the_image)] = vis
        coarse_dict[str(entity) + ' ' + str(coarse_label) + ' ' + str(in_the_image)] = vis
    return triplets, coarse_dict, idx, all_pred_sentence_number, all_wrong_match_sentence_number


def compute_f1_scores_quad(pred_pt, gold_pt, coarse_preds, coarse_labels):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    assert len(gold_pt) == len(pred_pt)
    none_dict = {'  ': []}
    for i in range(len(gold_pt)):

        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        if gold_pt[i] == none_dict:
            n_gold -= 1
        if pred_pt[i] == none_dict:
            n_gold -= 1

        for key in pred_pt[i]:
            if key in gold_pt[i] and ('False' in key or ('True' in key and pred_pt[i][key] in gold_pt[i][key])):
                n_tp += 1

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores


def compute_scores(pred_seqs, gold_seqs, vis_preds, img_labels, img_ids, object_detection_faults,
                   all_pred_sentence_number=0, all_wrong_match_sentence_number=0):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []
    all_coarse_labels, all_coarse_preds = [], []
    gold_idx, pred_idx = 0, 0

    # for i in range(num_samples):
    #     gold_list = extract_spans_para(gold_seqs[i], 'gold')
    #     pred_list = extract_spans_para(pred_seqs[i], 'pred')
    #     all_labels.append(gold_list)
    #     all_preds.append(pred_list)

    # print("\nResults:")
    # scores = compute_f1_scores(all_preds, all_labels)
    # print(scores)
    for i in range(num_samples):
        gold_dict, gold_coarse_dict, gold_idx, _, _ = extract_spans_para_quad(gold_seqs[i], img_labels, gold_idx,
                                                                              img_ids[i], object_detection_faults,
                                                                              'gold')
        pred_dict, pred_coarse_dict, pred_idx, all_pred_sentence_number, all_wrong_match_sentence_number = extract_spans_para_quad(
            pred_seqs[i], vis_preds, pred_idx, img_ids[i], object_detection_faults, 'pred', all_pred_sentence_number,
            all_wrong_match_sentence_number)

        all_labels.append(gold_dict)
        all_preds.append(pred_dict)

        all_coarse_labels.append(gold_coarse_dict)
        all_coarse_preds.append(pred_coarse_dict)

    print("\nResults:")
    scores = compute_f1_scores_quad(all_preds, all_labels, all_coarse_preds, all_coarse_labels)
    print(scores)

    print('\nall_pred_sentence_number = ', end='\t')
    print(all_pred_sentence_number)
    print('\nall_wrong_match_sentence_number = ', end='\t')
    print(all_wrong_match_sentence_number)

    return scores, all_labels, all_preds, all_coarse_labels, all_coarse_preds


def turn_vis_similarities_to_vis_pred(vis_similarities, outs, vinvl_region_number):
    """
    vis_similarities: [bts, max_length, vis_box_num]
    outs: [bts, max_length]
    """
    mask_for_classifier_index = []
    for pred in outs:
        _list_total = 0
        this_batch = []
        flag = True
        for token in pred:
            if token == 59:
                this_batch.append(False)
                this_batch.extend([False] * _list_total)
                _list_total = 0
                flag = False
                continue
            if token not in [16, 8, 1023]:  # the ids of "in the image"
                this_batch.append(False)
                this_batch.extend([False] * _list_total)
                _list_total = 0
            elif token == [16, 8, 1023][_list_total]:
                if not flag:
                    this_batch.append(False)
                    this_batch.extend([False] * _list_total)
                    _list_total = 0
                    flag = True
                    continue
                _list_total += 1
                if _list_total == 3:
                    this_batch.extend([True, True, True])
                    _list_total = 0
            else:
                this_batch.append(False)
                this_batch.extend([False] * _list_total)
                _list_total = 0
        this_batch.extend([False] * _list_total)
        mask_for_classifier_index.append(this_batch)

    mask_for_classifier_index = torch.tensor(mask_for_classifier_index)
    vis_similarities = vis_similarities[mask_for_classifier_index]
    if len(vis_similarities) == 0:
        return vis_similarities
    vis_pred = torch.argmax(vis_similarities.view(-1, 3, vinvl_region_number).mean(dim=1), dim=1)

    return vis_pred
