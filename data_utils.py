# -*- coding: utf-8 -*-

# This script contains all data transformation and reading

import random

import torch
import torchvision
from torch.utils.data import Dataset
from image_features_utils._image_features_reader import ImageFeaturesH5Reader  ##jmwang add
import numpy as np  ## jmwang add
import os
import json
import xml.etree.ElementTree as ET

coarse_fine_tree = {
    'location': ['city',
                 'country',
                 'state',
                 'continent',
                 'location_other',
                 'park',
                 'road'],
    'building': ['building_other',
                 'cultural_place',
                 'entertainment_place',
                 'sports_facility'],
    'organization': ['company',
                     'educational_institution',
                     'band',
                     'government_agency',
                     'news_agency',
                     'organization_other',
                     'political_party',
                     'social_organization',
                     'sports_league',
                     'sports_team'],
    'person': ['politician',
               'musician',
               'actor',
               'artist',
               'athlete',
               'author',
               'businessman',
               'character',
               'coach',
               'director',
               'intellectual',
               'journalist',
               'person_other'],
    'other': ['animal',
              'award',
              'medical_thing',
              'website',
              'ordinance'],
    'art': ['art_other',
            'film_and_television_works',
            'magazine',
            'music',
            'written_work'],
    'event': ['event_other',
              'festival',
              'sports_event'],
    'product': ['brand_name_products',
                'game',
                'product_other',
                'software']}


def fine2coarse_func(coarse_fine_tree=coarse_fine_tree):
    fine2coarse_map = {}
    for k, v in coarse_fine_tree.items():
        for t in v:
            fine2coarse_map[t] = k
    return fine2coarse_map


def read_image_label(image_annotation_path, img_id):
    fn = os.path.join(image_annotation_path, img_id + '.xml')

    tree = ET.parse(fn)
    root = tree.getroot()
    # aspect_box={}
    aspects = []
    boxes = []
    for object_container in root.findall('object'):
        for names in object_container.findall('name'):
            box_name = names.text
            box_container = object_container.findall('bndbox')
            if len(box_container) > 0:
                xmin = int(box_container[0].findall('xmin')[0].text)
                ymin = int(box_container[0].findall('ymin')[0].text)
                xmax = int(box_container[0].findall('xmax')[0].text)
                ymax = int(box_container[0].findall('ymax')[0].text)
            aspects.append(box_name)
            boxes.append([xmin, ymin, xmax, ymax])
    return aspects, boxes


def turn_labels_to_quad_and_get_visual_features(
        labels,
        img_ids,
        img_path_vinvl=None,
        image_annotation_path=None,
        vinvl_region_number=36,
        use_visual_feats=True
):
    new_label_list = []
    vis_feats = []
    vis_attention_masks = []
    object_detection_faults = {}

    for i, (label, img_id) in enumerate(zip(labels, img_ids)):

        image_num = 0
        image_tag = ''
        image_boxes = np.zeros((vinvl_region_number, 4), dtype=np.float32)
        image_feature = np.zeros((vinvl_region_number, 2048), dtype=np.float32)

        try:
            img = np.load(os.path.join(img_path_vinvl, str(img_id) + '.npz'))
        except:
            img = np.load(os.path.join(img_path_vinvl, '0.jpg.npz'))

        image_num = img['num_boxes']
        image_feature_ = img['box_features']
        # normalize
        image_feature_ = (image_feature_ / np.sqrt((image_feature_ ** 2).sum()))  ### 归一化   #### e.g.:0.000x

        final_num = min(image_num, vinvl_region_number)
        image_feature[:final_num] = image_feature_[:final_num]
        image_boxes[:final_num] = img['bounding_boxes'][:final_num]

        vis_attention_mask = [1] * int(final_num)
        if use_visual_feats is False:
            vis_attention_mask = [0] * int(final_num)

        vis_attention_mask.extend([0] * int(vinvl_region_number - final_num))

        vis_feats.append(image_feature)
        vis_attention_masks.append(vis_attention_mask)

        aspect_ious_dic = {}  # # {aspect:[iou1,iou2,...]}

        if os.path.isfile(os.path.join(image_annotation_path, img_id[:-4] + '.xml')):
            aspects, gt_boxes = read_image_label(image_annotation_path, img_id[:-4])
            try:
                IoUs = (torchvision.ops.box_iou(torch.tensor(gt_boxes).float(), torch.tensor(
                    image_boxes))).numpy()  # [x,4],[36,4]  ->[x,36] #! 版本的问题.float()
                for ith, a in enumerate(aspects):
                    cur_iou = IoUs[ith]
                    if max(cur_iou) < 0.5:  # # detector 没有检测到
                        # continue
                        # if a not in aspect_ious_dic:
                        #     object_detection_faults[img_id+' '+a] = True   # TODO
                        #     continue
                        # else:
                        #     object_detection_faults[img_id+' '+a] = False

                        if a not in aspect_ious_dic:  # # 首次出现这个name，赋 -1；如果之前已经有这个name的记录，无论任何记录，此处都不再关注
                            aspect_ious_dic[a] = np.array([
                                -1])  # # 如果在这里直接使 dict [key]= -1， 就导致 “一个实体标注多个框” 且 “任何一个没有被cover” ，就会覆盖掉之前的iou赋值，在下面assert报错，except直接pass。然后就发现不了错误
                    else:
                        if a in aspect_ious_dic:  # # 如果多个框，更新iou
                            last_iou = aspect_ious_dic[a]
                            if last_iou[0] == -1:  ## 之前的记录表示没有检测到
                                aspect_ious_dic[a] = cur_iou  ## 直接赋当前iou
                            else:
                                final_iou = np.array([max(last_iou[i], cur_iou[i]) for i in range(len(last_iou))])
                                aspect_ious_dic[a] = final_iou
                        else:
                            aspect_ious_dic[a] = cur_iou
            except:
                # print(image_name)
                pass
        else:
            pass

        quad = []
        object_detection_fault = {}
        for entity_tuple in label:
            e, fine_label, _, _ = entity_tuple
            is_in_image = False

            if e in aspect_ious_dic:
                ori_ious = aspect_ious_dic[e]
                if ori_ious[0] == -1:
                    average_iou = 0.
                    region_index = [average_iou] * (vinvl_region_number) + [1.]  ## 按照不相关训练
                    is_in_image = False
                    object_detection_fault[img_id + ' ' + e] = True
                else:
                    keeped_ious = np.array([iou if iou > 0.5 else 0 for iou in ori_ious])
                    norm_iou = keeped_ious / float(sum(keeped_ious))
                    region_index = norm_iou.tolist() + [0.]
                    is_in_image = True
            else:
                average_iou = 0.  # 1/box_num  # 0. # 1 / box_num
                region_index = [average_iou] * vinvl_region_number + [1.]
                is_in_image = False

            quad.append([e, fine_label, is_in_image, region_index])

        new_label_list.append(quad)
        object_detection_faults.update(object_detection_fault)

    return new_label_list, vis_feats, vis_attention_masks, object_detection_faults


def read_line_examples_from_file(data_path, img_path_vinvl=None, image_annotation_path=None, vinvl_region_number=None,
                                 use_visual_feats=True, silence=True):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels, img_ids = [], [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        for line in fp:
            line = line.strip()
            if line != '':
                words, entity_label, img_id = line.split('####')
                sents.append(words.split())
                img_ids.append(img_id)
                if entity_label[0] == '#':
                    entity_label = entity_label[1:]
                labels.append(eval(entity_label))
    if silence:
        print(f"Total examples = {len(sents)}")
    labels, vis_feats, vis_attention_masks, object_detection_faults = \
        turn_labels_to_quad_and_get_visual_features(
            labels,
            img_ids,
            img_path_vinvl=img_path_vinvl,
            image_annotation_path=image_annotation_path,
            vinvl_region_number=vinvl_region_number,
            use_visual_feats=use_visual_feats)
    return sents, labels, img_ids, vis_feats, vis_attention_masks, object_detection_faults


def get_para_targets(img_ids, labels, use_visual_feats=True, coarse_grained_auxiliary=True):
    targets = []
    if coarse_grained_auxiliary:
        fine2coarse_map = fine2coarse_func()

    for img_id, label in zip(img_ids, labels):
        all_sentences = []
        for quad in label:
            entity_name, fine_label, in_the_picture, _ = quad

            if entity_name == 'NULL':  # for implicit aspect term
                entity_name = 'it'
            if not use_visual_feats:
                one_sentence = f"{entity_name} is a {fine_label.lower()}"
            elif coarse_grained_auxiliary is False:  # use visual features without coarse_grained_auxiliary
                if in_the_picture:
                    one_sentence = f"{entity_name} is a {fine_label.lower()}, which is in the image"
                else:
                    # if img_id+' '+entity_name in object_detection_faults and object_detection_faults[img_id+' '+entity_name] is True:
                    #     one_sentence = f"{entity_name} is a {fine_label.lower()}, which is in the image"
                    # else:
                    one_sentence = f"{entity_name} is a {fine_label.lower()}, which is not in the image"

            else:

                c_label = fine2coarse_map[fine_label.lower()]

                if in_the_picture:
                    one_sentence = f"{entity_name} is a {c_label} and a {fine_label.lower()}, which is in the image"
                else:
                    # if img_id+' '+entity_name in object_detection_faults and object_detection_faults[img_id+' '+entity_name] is True:
                    #     one_sentence = f"{entity_name} is a {c_label} and a {fine_label.lower()}, which is in the image"
                    # else:
                    one_sentence = f"{entity_name} is a {c_label} and a {fine_label.lower()}, which is not in the image"

            all_sentences.append(one_sentence)

        target = ' [SSEP] '.join(all_sentences)
        # target = ' $ '.join(all_sentences)
        targets.append(target)
    return targets


def get_transformed_io(data_path=None, img_path_vinvl=None, image_annotation_path=None, vinvl_region_number=None,
                       use_visual_feats=None, coarse_grained_auxiliary=None, silence=True):
    """
    The main function to transform input & target according to the task
    """
    sents, labels, img_ids, vis_feats, vis_attention_masks, object_detection_faults = \
        read_line_examples_from_file(
            data_path,
            img_path_vinvl=img_path_vinvl,
            image_annotation_path=image_annotation_path,
            vinvl_region_number=vinvl_region_number,
            use_visual_feats=use_visual_feats,
            silence=silence)  # yl add img id
    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]

    # targets = get_para_targets(img_ids, labels, object_detection_faults, use_visual_feats=use_visual_feats, coarse_grained_auxiliary=coarse_grained_auxiliary)
    targets = get_para_targets(img_ids, labels, use_visual_feats=use_visual_feats,
                               coarse_grained_auxiliary=coarse_grained_auxiliary)
    img_labels = [[entity[-1][:-1] for entity in label] + [[0.0] * vinvl_region_number] * (6 - len(label)) for label in
                  labels]
    return inputs, targets, img_ids, vis_feats, vis_attention_masks, img_labels, object_detection_faults


class VisualDataset(Dataset):

    def __init__(self, tokenizer, data_dir, data_set, data_type, max_len=128, vinvl_region_number=36,
                 img_path_vinvl='/root/data2/twitter_images/twitterGMNER_vinvl_extract36',
                 image_annotation_path='/root/jmwang/brick/NER-CLS-VG/data/version2/xml', use_visual_feats=False,
                 coarse_grained_auxiliary=False):
        # './data/T5_data/train.txt'
        # self.has_caption = has_caption
        # self.data_path = f'data/{data_dir}/{data_type}.txt'

        self.data_path = f'{data_dir}/{data_type}.txt'

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_set

        self.inputs = []
        self.targets = []
        ## jmawng add

        self.vinvl_region_number = vinvl_region_number
        self.img_path_vinvl = img_path_vinvl
        self.use_visual_feats = use_visual_feats
        self.image_annotation_path = image_annotation_path
        self.coarse_grained_auxiliary = coarse_grained_auxiliary

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        vis_feat = self.vis_feats[index].squeeze()
        vis_mask = self.vis_attention_masks[index].squeeze()

        img_label = self.img_labels[index].squeeze()

        img_id = self.img_ids[index]

        return {"source_ids": source_ids, "source_mask": src_mask,
                "vis_feats": vis_feat, "vis_attention_mask": vis_mask,
                "target_ids": target_ids, "target_mask": target_mask,
                "img_label": img_label, "img_id": img_id}

    def _build_examples(self):
        inputs, targets, img_ids, vis_feats, vis_attention_masks, img_labels, object_detection_faults \
            = get_transformed_io(
            self.data_path,
            img_path_vinvl=self.img_path_vinvl,
            image_annotation_path=self.image_annotation_path,
            vinvl_region_number=self.vinvl_region_number,
            use_visual_feats=self.use_visual_feats,
            coarse_grained_auxiliary=self.coarse_grained_auxiliary,
            silence=True
        )

        for i in range(len(inputs)):
            text = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
                [text], max_length=self.max_len, padding="max_length",  # # jmwang add
                truncation=True, return_tensors="pt"
            )

            tokenized_target = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)

        self.img_ids = img_ids
        self.object_detection_faults = object_detection_faults
        self.vis_feats = torch.tensor(vis_feats)
        self.vis_attention_masks = torch.tensor(vis_attention_masks)
        self.img_labels = torch.tensor(img_labels)
