import os

image_annotation_path = '/root/data2/Twitter_Fine_Grained/version10000_final/xml'

path = '/root/data1/brick/FMNER_and_Grounding/KBS_0410/outputs/'

boxs = [24 - 4*i for i in range(5)]
lrs = [0.0003, 0.0002, 0.0001]

paths = []

for dirpath, dirnames, filenames in os.walk(path):
    # print(dirpath)
    # print(dirnames)
    # print(filenames)
    for file in filenames:
        if '.txt' in file:
            paths.append(os.path.join(dirpath, file))
    # for box in boxs:
    #     for lr in lrs:
    #         if str(box) + 'box' in dirpath and str(lr) in dirpath:
    #             for file in filenames:
    #                 if file.
    #             print(box, end='\t')
    #             print(lr)
                
# print(paths)

# breakpoint()

    
    
import os

coarse_fine_tree={
'location': [ 'city',
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
'organization': [ 'company',
              'educational_institution',
              'band',
              'government_agency',
              'news_agency',
              'organization_other',
              'political_party',
              'social_organization',
              'sports_league',
              'sports_team'],
'person': [ 'politician',
              'musician',
              'actor',
              'artist',
              'athlete',
              'author',
              'businessman',
              'character',
              'coach',
              'common_person',
              'director',
              'intellectual',
              'journalist',
              'person_other'],
'other': [ 'animal',
              'award',
              'medical_thing',
              'website',
              'ordinance'],
'art': [ 'art_other',
             'film_and_television_works',
             'magazine',
             'music',
             'written_work'],
'event': [ 'event_other',
             'festival',
             'sports_event'],
'product': [ 'brand_name_products',
              'game',
              'product_other',
              'software']}

fine_to_show = {
    'location': 'Location',
    'city': 'City',
    'country': 'Country',
    'state': 'State',
    'continent': 'Continent',
    'location_other': 'Location Other',
    'park': 'Park',
    'road': 'Road',
    'building': 'Building',
    'building_other': 'Building Other',
    'cultural_place': 'Cultural Place',
    'entertainment_place': 'Entertainment Place',
    'sports_facility': 'Sports Facility',
    'organization': 'Organization',
    'company': 'Company',
    'educational_institution': 'Educational Institution',
    'band': 'Band',
    'government_agency': 'Government Agency',
    'news_agency': 'News Agency',
    'organization_other': 'Organization Other',
    'political_party': 'Political Party',
    'social_organization': 'Social Organization',
    'sports_league': 'Sports League',
    'sports_team': 'Sports Team',
    'person': 'Person',
    'politician': 'Politician',
    'musician': 'Musician',
    'actor': 'Actor',
    'artist': 'Artist',
    'athlete': 'Athlete',
    'author': 'Author',
    'businessman': 'Businessman',
    'character': 'Character',
    'coach': 'Coach',
    'common_person': 'Common Person',
    'director': 'Director',
    'intellectual': 'Intellectual',
    'journalist': 'Journalist',
    'person_other': 'Person Other',
    'other': 'Other',
    'animal': 'Animal',
    'award': 'Award',
    'medical_thing': 'Medical Thing',
    'website': 'Website',
    'ordinance': 'Ordinance',
    'art': 'Art',
    'art_other': 'Art Other',
    'film_and_television_works': 'TV Works',
    'magazine': 'Magazine',
    'music': 'Music',
    'written_work': 'Written Work',
    'event': 'Event',
    'event_other': 'Event Other',
    'festival': 'Festival',
    'sports_event': 'Sports Event',
    'product': 'Product',
    'brand_name_products': 'Brand Name Products',
    'game': 'Game',
    'product_other': 'Product Other',
    'software': 'Software'
}


def fine2coarse_func(coarse_fine_tree=coarse_fine_tree):
    fine2coarse_map={}
    for k,v in coarse_fine_tree.items():
        for t in v:
            fine2coarse_map[t] = k
    return  fine2coarse_map
fine2coarse_map = fine2coarse_func()

def elaluate_fine_result(path=path):
    coarse_pred_dict = {}
    coarse_gt_dict = {}
    coarse_right_dict = {}

    for key in coarse_fine_tree:
        coarse_pred_dict[key] = 0
        coarse_gt_dict[key] = 0
        coarse_right_dict[key] = 0

    with open(path, 'r') as f:
        
        for line in f:
            line = line.strip()
            if len(line) == 0:
                for key in nowpred.keys():
                    if 'False' not in key and 'True' not in key:
                        break
                    
                    try:
                        fine_type, appear = key.split()[-2], key.split()[-1]
                    except:
                        breakpoint()

                    if fine_type not in fine2coarse_map:
                        break
                    coarse_type = fine2coarse_map[fine_type]
                    
                    coarse_pred_dict[coarse_type] += 1
                    if key in nowgt:
                        if 'False' in key or nowpred[key] in nowgt[key]:
                            coarse_right_dict[coarse_type] += 1
                            
                for key in nowgt.keys():
                    if 'False' not in key and 'True' not in key:
                        break
                    
                    
                    fine_type, appear = key.split()[-2], key.split()[-1]
                    coarse_type = fine2coarse_map[fine_type]  
                    coarse_gt_dict[coarse_type] += 1
                nowpred = {}
                nowgt = {}
                continue
            
            else:
                if line.startswith('fine     GT: '):
                    nowgt = eval(line.split('fine     GT: ')[-1])
                elif line.startswith('fine   Pred: '):
                    nowpred = eval(line.split('fine   Pred: ')[-1])

    # for key in coarse_pred_dict.keys():
    for key in ['person','location','building','organization','product','art','event','other']:
        # print(key, end='\t')
        p = 1.0 * coarse_right_dict[key] / coarse_gt_dict[key] * 100
        r = 1.0 * coarse_right_dict[key] / coarse_pred_dict[key] * 100
        
        f1 =  2.0 * p * r / (p + r)
        # print(f"p = {p:.4} ; r = {r:.4} ; f1 = {f1:.4}")
        print(f'{f1:.4}', end ='\t&')
        
# elaluate_fine_result()


def elaluate_all(path=path):
    
    gt_num = 0
    pred_num = 0
    right = 0
    right1, right2 = 0, 0

    with open(path, 'r') as f:
        
        for line in f:
            line = line.strip()
            if len(line) == 0:
                for key in nowpred.keys():
                    macthed = False
                    if 'False' not in key and 'True' not in key:
                        continue
                    pred_num += 1
                    
                    # entity, fine_type, appear = key.split()[:-2], key.split()[-2], key.split()[-1]
                    # entity = ' '.join(entity)
                    if key in nowgt.keys():
                        if 'False' in key:
                            right1 += 1
                        elif nowpred[key] in nowgt[key]:
                            right2 += 1
                        
                    # for gt_key in nowgt.keys():
                    #     gt_entity, _, _ = key.split()[:-2], key.split()[-2], key.split()[-1]
                    #     gt_entity = ' '.join(gt_entity)
                    #     if entity == gt_entity and appear in gt_key:

                    #         if appear == 'False':
                    #             macthed = True
                    #             right1 += 1
                    #         elif nowpred[key] in nowgt[gt_key]:
                    #             macthed = True
                    #             right2 += 1
                                
                    #     if macthed:
                    #         break
                    
                    # if macthed:
                    #     right += 1
                        
                for key in nowgt.keys():
                    if 'False' not in key and 'True' not in key:
                        continue
                    gt_num += 1
                    
                    
                nowpred = {}
                nowgt = {}
                continue
            
            else:
                if line.startswith('fine     GT: '):
                    nowgt = eval(line.split('fine     GT: ')[-1])
                    # gt_num += len(nowgt)
                elif line.startswith('fine   Pred: '):
                    nowpred = eval(line.split('fine   Pred: ')[-1])
                    # pred_num += len(nowpred)
                    

    right = right1 + right2
    p = 1.0 * right / gt_num * 100
    r = 1.0 * right / pred_num * 100

    f1 =  2.0 * p * r / (p + r)

    # print(f'no_box_right:{right1}')
    # print(f'have_box_right:{right2}')
    
    # print(f'all right:{right}')
    # print(f'gt_num:{gt_num}')
    # print(f'pred_num:{pred_num}')
 
    print(f"{p:.4}\t{r:.4}\t{f1:.4}", end="\t")

# elaluate_all()


def eval_ouput_file_FMNER(path=path):
    
    pred_pt = []
    gold_pt = []
    
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if s.startswith('fine     GT: '):
                gold_pt.append(eval(s.replace('fine     GT: ', '').replace(' False', '').replace(' True', '')))
            elif s.startswith('fine   Pred: '):
                pred_pt.append(eval(s.replace('fine   Pred: ', '').replace(' False', '').replace(' True', '')))
            else:
                continue
            
    n_tp, n_gold, n_pred = 0, 0, 0

    assert len(gold_pt) == len(pred_pt)
    
    for i in range(len(gold_pt)):
        # n_gold += len(gold_pt[i])
        # n_pred += len(pred_pt[i])
        for key in pred_pt[i]:
            if len(key.strip()) == 0:
                continue
            n_pred += 1
            if key in gold_pt[i]:
                n_tp += 1
        for key in gold_pt[i]:
            if len(key.strip()) == 0:
                continue
            n_gold += 1

    # print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) * 100 if n_pred != 0 else 0 
    recall = float(n_tp) / float(n_gold) * 100 if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    # scores = {'precision': precision, 'recall': recall, 'f1': f1}

    print(f'{precision:.4}\t{recall:.4}\t{f1:.4}', end="\t")
                
# eval_ouput_file_FMNER()         
                 
def elaluate_EEG(path=path):
    
    gt_num = 0
    pred_num = 0
    right = 0
    right1, right2 = 0, 0
    wrong_no_box = 0
    wrong_have_box = 0
    gt_true, gt_false = 0, 0
    pred_false, pred_true = 0, 0
    all_right = 0

    with open(path, 'r') as f:
        
        for line in f:
            line = line.strip()
            if len(line) == 0:
                for key in nowpred.keys():
                    macthed = False
                    if 'False' not in key and 'True' not in key:
                        continue
                    
                    pred_num += 1
                    
                    if 'False' in key:
                        pred_false += 1
                    if 'True' in key:
                        pred_true += 1
                        
                        
                    entity, fine_type, appear = key.split()[:-2], key.split()[-2], key.split()[-1]
                    entity = ' '.join(entity)
                        
                    for gt_key in nowgt.keys():
                        if 'False' not in gt_key and 'True' not in gt_key:
                            continue
                        gt_entity, _, _ = gt_key.split()[:-2], gt_key.split()[-2], gt_key.split()[-1]
                        gt_entity = ' '.join(gt_entity)
                        if entity == gt_entity and appear in gt_key:

                            if appear == 'False':
                                macthed = True
                                right1 += 1
                            elif nowpred[key] in nowgt[gt_key]:
                                macthed = True
                                right2 += 1
                        # elif entity == gt_entity and appear == 'False':
                        #     wrong_no_box += 1
                            
                        # if entity == gt_entity and 'False' in gt_key:
                        #     all_note_no_box += 1
                            
                        if macthed:
                            # breakpoint()
                            # print(key +'\t'+ gt_key)
                            if key == gt_key:
                                all_right += 1
                            break
                        
                    
                    if macthed:
                        right += 1
                    else:
                        if appear == 'True':
                            wrong_have_box += 1
                        elif appear == 'False':
                            wrong_no_box += 1
                        
                for key in nowgt.keys():
                    if 'False' not in key and 'True' not in key:
                        continue
                        
                    if 'True' in key:
                        gt_true += 1
                    else:
                        gt_false += 1
                    gt_num += 1
                    
                nowpred = {}
                nowgt = {}
                continue
            
            else:
                if line.startswith('fine     GT: '):
                    nowgt = eval(line.split('fine     GT: ')[-1])
                    # gt_num += len(nowgt)
                elif line.startswith('fine   Pred: '):
                    nowpred = eval(line.split('fine   Pred: ')[-1])
                    # pred_num += len(nowpred)
                    
    # right = all_note_no_box
    
    p = 1.0 * right / gt_num * 100
    r = 1.0 * right / pred_num * 100

    f1 =  2.0 * p * r / (p + r)
    # print(f'no_box_right\t\t{right1}')
    # print(f'have_box_right\t\t{right2}')
    
    # print(f'all right\t\t{right}')
    
    # print(f'pred_num\t\t{pred_num}')
    # print(f'pred_true\t\t{pred_true}')
    # print(f'wrong_have_box\t\t{wrong_have_box}')
    # print(f'pred_false\t\t{pred_false}')
    # print(f'wrong_no_box\t\t{wrong_no_box}')
    
    # print(f'gt_num\t\t\t{gt_num}')
    # print(f'gt_true\t\t\t{gt_true}')
    # print(f'gt_false\t\t{gt_false}')
    
    # print(f'all_right\t\t{all_right}')
    
    
    
    print(f"{p:.4}\t{r:.4}\t{f1:.4}")

# elaluate_EEG()
import xml.etree.ElementTree as ET          
def read_image_label(image_annotation_path='/root/data2/Twitter_Fine_Grained/version10000_final/xml',img_id=None):
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


# read_image_label(img_id='O_1659.jpg')

def sunburst_print():

    data_path = './T5_data'

    sum_fine = {key: 0 for key in fine2coarse_map}
    sum_coarse = {key: 0 for key in coarse_fine_tree}
    
    groundable_fine = {key: 0 for key in fine2coarse_map}
    
    sum_all = 0
    groundable_num = 0
    groundable_set = 0
    annotation_box_num = 0
    # data = {
    #     'fine' : [],
    #     'coarse' : [],
    # }
    for mode in ['test.txt', 'dev.txt', 'train.txt']:
    # for mode in ['train.txt']:
        with open(os.path.join(data_path, mode), 'r', encoding='UTF-8') as fp:
            for line in fp:
                line = line.strip()
                if line != '':
                    words, entity_label, img_id = line.split('####')
                    if entity_label[0] == '#':
                        entity_label = entity_label[1:]
                    labels = eval(entity_label)
                    
                    aspects = []
                    gt_boxes = []
                    if os.path.isfile(os.path.join(image_annotation_path, img_id[:-4] + '.xml')):

                        aspects, gt_boxes = read_image_label(image_annotation_path, img_id[:-4])
                        groundable_num += len(set(aspects))
                        annotation_box_num += len(gt_boxes)
                    for label in labels:
                        if label[1] == 'common_person':
                            label[1] = 'person_other'
                            # breakpoint()
                        sum_fine[label[1]] += 1
                        sum_coarse[fine2coarse_map[label[1]]] += 1
                        sum_all += 1
                        if label[0] in aspects:
                            try:
                                groundable_fine[label[1]] += 1
                            except:
                                breakpoint()

                        # data['fine'].append(label[1])
                        # data['coarse'].append(fine2coarse_map[label[1]])
                        
                    
                        # breakpoint()

    # breakpoint()
    coarse_persent = {key: f'{(sum_coarse[key] / sum_all * 100) :.4}% ' for key in sum_coarse.keys()}
    coarse_persent = {key: format((sum_coarse[key] / sum_all * 100),'0.2f') +'%' for key in sum_coarse.keys()}
    # keys = [key for key in sum_fine.keys()]
    sorted_tuples = sorted(sum_fine.items(),  key=lambda d: d[1], reverse=True)
    keys = [w[0] for w in sorted_tuples][::-1]
    # print(keys)
    data = {
        'fine':keys,
        # 'coarse':[fine2coarse_map[key]+str('\n'+coarse_persent[fine2coarse_map[key]]) for key in keys],
        'coarse':[fine2coarse_map[key] for key in keys],
        'value':[sum_fine[key] for key in keys],
        'groundable_value':[groundable_fine[key] for key in keys]
    }
    # print(groundable_num)
    # print(annotation_box_num)
    print(data)
            
    import pandas as pd
    df = pd.DataFrame(data)
    print(df)
    # # data=df
    # color_discrete_sequence = ['#ee7959','#fbca4d', '#ffee6f','#a0d8ef','#c5e1a5','#dbb6d6','#b2ebf2','#f5f2e9']
    # # sunburst_print
    # import plotly.express as px
    # from plotly.subplots import make_subplots
    # import plotly.graph_objects as go
    # fig1=px.sunburst(data, path=['coarse','fine'],values='value', color_discrete_sequence=color_discrete_sequence)
    # # fig=px.sunburst(data, path=['coarse','fine'],values='value', color_discrete_sequence=px.colors.qualitative.Alphabet)
    # fig2 = make_subplots(rows=1, cols=2,column_width=[0.4,0.6],
    #                      specs=[[{"type": "domain"}, {"type": "xy"}]])
    # # breakpoint()
    
    # coarse_keys = list(set(data['coarse']))
    

    
    # labels = data['fine']+coarse_keys
    # parents = [fine2coarse_map[label] if label in fine2coarse_map else '' for label in labels]
    # values = data['value'] + [0 for i in labels[len(data['value']):]]
    # # breakpoint()
    # # colors = []
    # # for label in labels:
    # #     if label in coarse_keys:
    # #         colors.append(color_discrete_sequence[coarse_keys.index(label)])
    # #         continue
    # #     if label in fine2coarse_map:
    # #         label = fine2coarse_map[label]
    # #         colors.append(color_discrete_sequence[coarse_keys.index(label)])
    # #         continue
    #     # breakpoint()
    # colors = [color_discrete_sequence[coarse_keys.index(key)] if key in coarse_keys else color_discrete_sequence[coarse_keys.index(fine2coarse_map[key])] for key in labels]
    # fig2.add_trace(go.Sunburst(
    #     labels=labels, 
    #     parents=parents,
    #     values=values,
    #     # insidetextorientation='radial',
    #     marker=dict(colors=colors)
    #     ),row=1, col=1)
    # # fig2 = go.Figure()
    # fig2.add_trace(go.Bar(
    #     y=data['fine'], 
    #     x=data['groundable_value'],
    #     orientation='h',
    #     marker=dict(
    #         color='#ff85c0',
    #         line=dict(color='rgb(248, 248, 249)', width=1)
    #         ),
    #     name='Groundable'),
    #                row=1,col=2
    #                )
    # fig2.add_trace(go.Bar(
    #     y=data['fine'], 
    #     x=data['value'],
    #     orientation='h',
    #     marker=dict(
    #         color='#8c8c8c',
    #         line=dict(color='rgb(248, 248, 249)', width=1)
    #         ),
    #     name='Not Groundable'
    #     ),
    #                row=1,col=2
    #                )
    # fig2.update_layout(height=500,width=1200,barmode='stack')
    # # fig.update_layout(barmode='stack')
    # # fig2.update_traces(insidetextorientation='radial')
    # # fig2.update_layout(margin = dict(t=0, l=0, r=0, b=0))
    # # fig2.show()
    # # fig2=px.bar(data,y='fine', x='value',orientation='h')
    # fig2.write_image("fig5.png", scale=10)
    
    
# sunburst_print()


# import os

# # 指定目标目录
# path = '/root/jmwang/brick/NER-CLS-VG/data/version10000/xml'

# # 获取目录中所有文件和目录的列表
# file_list = os.listdir(path)

# # 计算文件个数
# file_count = len(file_list)

# print(f'The directory "{path}" contains {file_count} files.')

def calculate_all_path():
    boxs = [36 - 4*i for i in range(9)]
    lrs = [0.0001, 0.0002, 0.0003, 0.0005]
    # print(boxs)
    paths = ['/root/data1/brick/FMNER_and_Grounding/KBS_0410/outputs/_SSEP_10epoch_32box_42_0.0002_20230725-13:03:41/results-0.0002-32-42.txt']
    
    for path in paths:
        # print(path)
        flag = False
        for box in boxs:

            if str('_')+ str(box) + 'box' in path:
                for lr in lrs:
                    if str(lr) in path:
                        print(box, end='\t')
                        print(lr, end='\t')
                        flag = True
                        break
                    
        if flag is False:
            continue
        else:
            flag = False
        elaluate_all(path)
        eval_ouput_file_FMNER(path)
        elaluate_EEG(path)
        elaluate_fine_result(path)


calculate_all_path()

path = '/root/data1/brick/FMNER_and_Grounding/KBS_0410/outputs/_SSEP_10epoch_32box_42_0.0003_20230726-12:27:24/results-0.0003-32-42.txt'
def calculate_coarse(path=path):
    gt_num = 0
    pred_num = 0
    right = 0
    right1, right2 = 0, 0

    with open(path, 'r') as f:
        
        for line in f:
            line = line.strip()
            if len(line) == 0:
                for key in nowpred.keys():
                    macthed = False
                    if 'False' not in key and 'True' not in key:
                        continue
                    pred_num += 1
                    
                    # entity, fine_type, appear = key.split()[:-2], key.split()[-2], key.split()[-1]
                    # entity = ' '.join(entity)
                    if key in nowgt.keys():
                        if 'False' in key:
                            right1 += 1
                        elif nowpred[key] in nowgt[key]:
                            right2 += 1
                        
                    # for gt_key in nowgt.keys():
                    #     gt_entity, _, _ = key.split()[:-2], key.split()[-2], key.split()[-1]
                    #     gt_entity = ' '.join(gt_entity)
                    #     if entity == gt_entity and appear in gt_key:

                    #         if appear == 'False':
                    #             macthed = True
                    #             right1 += 1
                    #         elif nowpred[key] in nowgt[gt_key]:
                    #             macthed = True
                    #             right2 += 1
                                
                    #     if macthed:
                    #         break
                    
                    # if macthed:
                    #     right += 1
                        
                for key in nowgt.keys():
                    if 'False' not in key and 'True' not in key:
                        continue
                    gt_num += 1
                    
                    
                nowpred = {}
                nowgt = {}
                continue
            
            else:
                if line.startswith('coarse   GT: '):
                    nowgt = eval(line.split('coarse   GT: ')[-1])
                    # gt_num += len(nowgt)
                elif line.startswith('coarse Pred: '):
                    nowpred = eval(line.split('coarse Pred: ')[-1])
                    # pred_num += len(nowpred)
                    

    right = right1 + right2
    p = 1.0 * right / gt_num * 100
    r = 1.0 * right / pred_num * 100

    f1 =  2.0 * p * r / (p + r)

    # print(f'no_box_right:{right1}')
    # print(f'have_box_right:{right2}')
    
    # print(f'all right:{right}')
    # print(f'gt_num:{gt_num}')
    # print(f'pred_num:{pred_num}')
 
    print(f"{p:.4}\t{r:.4}\t{f1:.4}", end="\t")
    
# calculate_coarse()



def check_the_relevance(path=path):
    coarse_pred = {}
    fine_pred = {}
    no_seen = 0
    wrong_reflection = 0
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                if coarse_pred == {'  ': []}:
                    continue
                if len(coarse_pred) and len(fine_pred):
                    for key in fine_pred:
                        entity, fine_type, appear = key.split()[:-2], key.split()[-2], key.split()[-1] 
                        try:
                            coarse_key = ' '.join(entity) +' '+ str(fine2coarse_map[fine_type]) +' '+ appear
                        except:
                            coarse_key = None
                        if coarse_key is None:
                            no_seen += 1
                            breakpoint()
                            continue
                        if coarse_key not in coarse_pred:
                            wrong_reflection += 1
                            breakpoint()
                coarse_pred = {}
                fine_pred = {}
                continue
            if line.startswith("coarse Pred:"):
                coarse_pred = eval(line.split('coarse Pred: ')[-1])
            if line.startswith('fine   Pred:'):
                fine_pred = eval(line.split('fine   Pred: ')[-1])
    
    print(f'no_seen = {no_seen}')
    print(f'wrong_reflection = {wrong_reflection}')
                
# check_the_relevance()



def case_choose(TIGER_path=path, ITA_path='/root/jmwang/brick/FMNERG/step2/eval_ent_ty/2023-04-20-17-20-17_version10000_0.3_20/triple_pred.txt',MMT5_path='/root/jmwang/brick/FMNERG/step2/eval_ent_ty/2023-04-20-17-02-53_version10000_0.3_20/triple_pred.txt'):
    
    with open(TIGER_path, 'r') as f:
        gt = {}
        pred = {}
        
        for line in f:
            if line.startswith('fine     GT:'):
                gt = eval(line.split('fine     GT: ')[-1])
            if line.startswith('fine   Pred:'):
                pred = eval(line.split('fine   Pred: ')[-1])
            if len(gt) and len(pred):
                if gt == {'  ': []}:
                    gt = ''
                    pred = ''
                    continue
                # if len(gt) == len(pred):
                #     gt = ''
                #     pred = ''
                #     continue
                count = 0
                for key in pred:
                    if key in gt:
                        if 'True' in key:
                            count += 1
                        # if 'True' in key and pred[key] in gt[key]:
                        # if pred[key] in gt[key]:
                        #     count += 1
                        #     # print(f'  gt={gt}')
                        #     # print(f'pred={pred}')
                        #     # print()
                        #     break
                
                if count < len(gt):
                    print(f'  gt={gt}')
                    print(f'pred={pred}')
                    print()
                    

                gt = ''
                pred = ''

# case_choose()


import numpy as np
def read_vinVL(img_path_vinvl = '/root/data2/twitter_images/twitterGMNER_vinvl_extract36'):
    img_id = 'O_1659.jpg'
    img = np.load(os.path.join(img_path_vinvl, str(img_id) + '.npz'))
    breakpoint()
    
# read_vinVL()