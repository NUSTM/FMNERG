import os

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# image_annotation_path = '/root/jmwang/brick/NER-CLS-VG/data/version10000/xml'

# coarse_fine_tree={
# 'location': [ 'city',
#  			'country',
#  			'state',
#  			'continent',
# 			'location_other',
#  			'park',
#  			'road'],
# 'building': ['building_other',
#  			'cultural_place',
#  			'entertainment', #'entertainment_place',
#  			'sports_facility'],
# 'organization': [ 'company',
# 			  'educational_institution',
# 			  'band',
# 			  'government_agency',
# 			  'news_agency',
# 			  'organization_other',
# 			  'political_party',
# 			  'social_organization',
# 			  'sports_league',
# 			  'sports_team'],
# 'person': [ 'politician',
# 			  'musician',
# 			  'actor',
# 			  'artist',
# 			  'athlete',
# 			  'author',
# 			  'businessman',
# 			  'character',
# 			  'coach',
# 			  'common_person',
# 			  'director',
# 			  'intellectual',
# 			  'journalist',
# 			  'person_other'],
# 'other': [ 'animal',
# 			  'award',
# 			  'medical_thing',
# 			  'website',
# 			  'ordinance'],
# 'art': [ 'art_other',
# 			#  'film_and_television_works',
#             'TV works',
# 			 'magazine',
# 			 'music',
# 			 'written_work'],
# 'event': [ 'event_other',
# 			 'festival',
# 			 'sports_event'],
# 'product': [ 'brand_name_products',
# 			  'game',
# 			  'product_other',
# 			  'software']}

coarse_fine_tree = {
    'Location': ['City',
                 'Country',
                 'State',
                 'Continent',
                 'Location Other',
                 'Park',
                 'Road'],
    'Building': ['Building Other',
                 'Cultural Place',
                 'Entertainment',  # 'Entertainment Place',
                 'Sports Facility'],
    'Organization': ['Company',
                     'Educational Institution',
                     'Band',
                     'Government Agency',
                     'News Agency',
                     'Organization Other',
                     'Political Party',
                     'Social Organization',
                     'Sports League',
                     'Sports Team'],
    'Person': ['Politician',
               'Musician',
               'Actor',
               'Artist',
               'Athlete',
               'Author',
               'Businessman',
               'Character',
               'Coach',
               'Common Person',
               'Director',
               'Intellectual',
               'Journalist',
               'Person Other'],
    'Other': ['Animal',
              'Award',
              'Medical Thing',
              'Website',
              'Ordinance'],
    'Art': ['Art Other',
            # 'Film And Television Works',
            'TV Works',
            'Magazine',
            'Music',
            'Written Work'],
    'Event': ['Event Other',
              'Festival',
              'Sports Event'],
    'Product': ['Brand Name Products',
                'Game',
                'Product Other',
                'Software']}


def fine2coarse_func(coarse_fine_tree):
    fine2coarse_map = {}
    for k, v in coarse_fine_tree.items():
        for t in v:
            fine2coarse_map[t] = k
    return fine2coarse_map


fine2coarse_map = fine2coarse_func(coarse_fine_tree)


def sunburst_print():
    """
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
    """

    ###
    data = {
        'fine': ['Continent', 'Medical Thing', 'Art Other', 'Ordinance', 'Magazine', 'Game', 'Written Work', 'Event Other',
                 'Organization Other', 'Road', 'Cultural Place', 'Director', 'Political Party', 'Sports Facility',
                 'Journalist', 'Animal', 'Coach', 'Intellectual', 'Music', 'Building Other', 'Software', 'Award',
                 'Brand Name Products', 'Website', 'Government Agency', 'State', 'Product Other', 'Social Organization',
                 'Sports Event', 'Author', 'Entertainment', 'Park', 'Person Other', 'Educational Institution',
                 'Businessman', 'Band', 'Character', 'News Agency', 'Location Other', 'TV Works',
                 'Country', 'Artist', 'Festival', 'Company', 'Sports League', 'City', 'Actor', 'Politician', 'Musician',
                 'Sports Team', 'Athlete'],
        'coarse': ['Location', 'Other', 'Art', 'Other', 'Art', 'Product', 'Art', 'Event', 'Organization', 'Location',
                   'Building', 'Person', 'Organization', 'Building', 'Person', 'Other', 'Person', 'Person', 'Art',
                   'Building', 'Product', 'Other', 'Product', 'Other', 'Organization', 'Location', 'Product',
                   'Organization', 'Event', 'Person', 'Building', 'Location', 'Person', 'Organization', 'Person',
                   'Organization', 'Person', 'Organization', 'Location', 'Art', 'Location', 'Person', 'Event',
                   'Organization', 'Organization', 'Location', 'Person', 'Person', 'Person', 'Organization', 'Person'],
        'value': [34, 37, 37, 39, 49, 56, 60, 66, 68, 70, 73, 75, 84, 86, 108, 111, 113, 115, 116, 122, 127, 135, 147, 160,
                  169, 189, 195, 196, 197, 219, 223, 226, 228, 233, 259, 262, 268, 270, 329, 442, 504, 580, 606, 665, 707,
                  873, 934, 1225, 1310, 1614, 1761],
        'groundable_value': [3, 2, 27, 5, 23, 18, 27, 5, 7, 8, 16, 46, 11, 28, 56, 95, 82, 66, 23, 32, 32, 24, 78, 22, 16,
                             7, 98, 29, 35, 68, 30, 27, 129, 37, 161, 137, 169, 55, 57, 58, 30, 198, 119, 162, 98, 12, 745,
                             729, 927, 602, 1299]
    }


    
    for i in range(len(data['fine'])):
        data['value'][i] -= data['groundable_value'][i]
    # print(groundable_num)
    # print(annotation_box_num)

    # import pandas as pd
    # df = pd.DataFrame(data)

    color_discrete_sequence = ['#ee7959', '#fbca4d', '#ffee6f', '#a0d8ef', '#c5e1a5', '#dbb6d6', '#b2ebf2', '#f5f2e9']
    # sunburst_print

    fig1 = px.sunburst(data, path=['coarse', 'fine'], values='value', color_discrete_sequence=color_discrete_sequence)
    # fig=px.sunburst(data, path=['coarse','fine'],values='value', color_discrete_sequence=px.colors.qualitative.Alphabet)

    fig2 = make_subplots(rows=1, cols=2, column_width=[0.4, 0.6],  # column_width=[0.4,0.6],
                         specs=[[{"type": "domain"}, {"type": "xy"}]])

    coarse_keys = list(set(data['coarse']))
    labels = data['fine'] + coarse_keys
    parents = [fine2coarse_map[label] if label in fine2coarse_map else '' for label in labels]
    values = data['value'] + [0 for i in labels[len(data['value']):]]

    colors = [color_discrete_sequence[coarse_keys.index(key)] if key in coarse_keys else color_discrete_sequence[
        coarse_keys.index(fine2coarse_map[key])] for key in labels]
    fig2.add_trace(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        # insidetextorientation='radial',
        marker=dict(colors=colors)
    ), row=1, col=1)
    # fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        y=data['fine'][32:],  ## jmwang
        x=data['groundable_value'][32:],  ## jmwang
        orientation='h',
        marker=dict(
            color='#ff85c0',
            line=dict(color='rgb(248, 248, 249)', width=1)
        ),
        name='Groundable'),
        row=1, col=2
    )
    fig2.add_trace(go.Bar(
        y=data['fine'][32:],  ##jmwamg
        x=data['value'][32:],  ##jmwamg
        orientation='h',
        marker=dict(
            color='#8c8c8c',
            line=dict(color='rgb(248, 248, 249)', width=1)
        ),
        name='Not Groundable'
    ),
        row=1, col=2
    )
    fig2.update_layout(height=500, width=1200, barmode='stack',
                       legend=dict(  ##### fig2 legend jmwang
                           yanchor="bottom",
                           y=0.01,
                           xanchor="right",
                           x=0.99,
                           title=dict(text=''),
                           # font=dict(size = 20)
                       ))

    # fig.update_layout(barmode='stack')
    # fig2.update_traces(insidetextorientation='radial')
    # fig2.update_layout(margin = dict(t=0, l=0, r=0, b=0))
    # fig2.show()
    # fig2=px.bar(data,y='fine', x='value',orientation='h')
    fig2.write_image("fig.png", scale=1)


def sunburst_only():
    ###
    data = {
        'fine': ['Continent', 'Medical Thing', 'Art Other', 'Ordinance', 'Magazine', 'Game', 'Written Work', 'Event Other',
                 'Organization Other', 'Road', 'Cultural Place', 'Director', 'Political Party', 'Sports Facility',
                 'Journalist', 'Animal', 'Coach', 'Intellectual', 'Music', 'Building Other', 'Software', 'Award',
                 'Brand Name Products', 'Website', 'Government Agency', 'State', 'Product Other', 'Social Organization',
                 'Sports Event', 'Author', 'Entertainment', 'Park', 'Person Other', 'Educational Institution',
                 'Businessman', 'Band', 'Character', 'News Agency', 'Location Other', 'TV Works',
                 'Country', 'Artist', 'Festival', 'Company', 'Sports League', 'City', 'Actor', 'Politician', 'Musician',
                 'Sports Team', 'Athlete'],
        'coarse': ['Location', 'Other', 'Art', 'Other', 'Art', 'Product', 'Art', 'Event', 'Organization', 'Location',
                   'Building', 'Person', 'Organization', 'Building', 'Person', 'Other', 'Person', 'Person', 'Art',
                   'Building', 'Product', 'Other', 'Product', 'Other', 'Organization', 'Location', 'Product',
                   'Organization', 'Event', 'Person', 'Building', 'Location', 'Person', 'Organization', 'Person',
                   'Organization', 'Person', 'Organization', 'Location', 'Art', 'Location', 'Person', 'Event',
                   'Organization', 'Organization', 'Location', 'Person', 'Person', 'Person', 'Organization', 'Person'],
        'value': [34, 37, 37, 39, 49, 56, 60, 66, 68, 70, 73, 75, 84, 86, 108, 111, 113, 115, 116, 122, 127, 135, 147, 160,
                  169, 189, 195, 196, 197, 219, 223, 226, 228, 233, 259, 262, 268, 270, 329, 442, 504, 580, 606, 665, 707,
                  873, 934, 1225, 1310, 1614, 1761],
        'groundable_value': [3, 2, 27, 5, 23, 18, 27, 5, 7, 8, 16, 46, 11, 28, 56, 95, 82, 66, 23, 32, 32, 24, 78, 22, 16,
                             7, 98, 29, 35, 68, 30, 27, 129, 37, 161, 137, 169, 55, 57, 58, 30, 198, 119, 162, 98, 12, 745,
                             729, 927, 602, 1299]
    }



    # for i in range(len(data['fine'])):
    #     data['value'][i] -= data['groundable_value'][i]

    color_discrete_sequence = [ '#fbca4d', '#ffee6f', '#a0d8ef','#ee7959', '#c5e1a5', '#dbb6d6', '#b2ebf2', '#f5f2e9']
    color_discrete_sequence = [ '#8fd3c6', '#bfb9d9', '#7bb4d2', '#fb8071','#b3de6a', '#bc80bc','#fcb462', '#ffea6c']
    color_discrete_sequence = [ '#b7e6de', '#d8d5e8', '#b6cfe3', '#b6cfe3','#d2eca5', '#d7b2d7','#fed2a1', '#fff4a8']
    color_discrete_sequence = [ '#80deea', '#e1bee7', '#a5d6a7', '#e6ee9c','#ffe082', '#f8bbd0','#ffab91', '#90caf9']
    # sunburst_print

    fig2 = make_subplots(rows=1, cols=2, column_width=[1, 0],  # column_width=[0.4,0.6],
                         specs=[[{"type": "domain"}, {"type": "xy"}]]
                         )
    # breakpoint()

    coarse_keys = list(set(data['coarse']))
    coarse_keys = sorted(coarse_keys)
    print(coarse_keys)
    labels = data['fine'] + coarse_keys
    parents = [fine2coarse_map[label] if label in fine2coarse_map else '' for label in labels]
    values = data['value'] + [0 for i in labels[len(data['value']):]]
    # breakpoint()
    colors = [color_discrete_sequence[coarse_keys.index(key)] if key in coarse_keys else color_discrete_sequence[coarse_keys.index(fine2coarse_map[key])] for key in labels]
    fig2.add_trace(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        insidetextorientation='radial',
        marker=dict(colors=colors)
    ), row=1, col=1)
    # fig2 = go.Figure()
    # fig2.add_trace(go.Bar(
    #     y=data['fine'][32:], ## jmwang
    #     x=data['groundable_value'][32:], ## jmwang
    #     orientation='h',
    #     marker=dict(
    #         color='#ff85c0',
    #         line=dict(color='rgb(248, 248, 249)', width=1)
    #         ),
    #     name='Groundable'),
    #                row=1,col=2
    #                )
    # fig2.add_trace(go.Bar(
    #     y=data['fine'][32:], ##jmwamg
    #     x=data['value'][32:], ##jmwamg
    #     orientation='h',
    #     marker=dict(
    #         color='#8c8c8c',
    #         line=dict(color='rgb(248, 248, 249)', width=1)
    #         ),
    #     name='Not Groundable'
    #     ),
    #                row=1,col=2
    #                )
    fig2.update_layout(height=500, width=500, barmode='stack',
                       # legend=dict(  ##### fig2 legend jmwang
                       #     yanchor="bottom",
                       #     y=0.01,
                       #     xanchor="right",
                       #     x=0.99,
                       #     title=dict(text =''),
                       #     # font=dict(size = 20)
                       # )
                       )

    # fig.update_layout(barmode='stack')
    # fig2.update_traces(insidetextorientation='radial')
    # fig2.update_layout(margin = dict(t=0, l=0, r=0, b=0))
    # fig2.show()
    # fig2=px.bar(data,y='fine', x='value',orientation='h')
    fig2.write_image("fig_sunonly.pdf", scale=10, format='pdf')
    fig2.write_image("fig_sunonly.png", scale=5)


def bar_only():
    ###
    data = {
        'fine': ['Continent', 'Medical Thing', 'Art Other', 'Ordinance', 'Magazine', 'Game', 'Written Work', 'Event Other',
                 'Organization Other', 'Road', 'Cultural Place', 'Director', 'Political Party', 'Sports Facility',
                 'Journalist', 'Animal', 'Coach', 'Intellectual', 'Music', 'Building Other', 'Software', 'Award',
                 'Brand Name Products', 'Website', 'Government Agency', 'State', 'Product Other', 'Social Organization',
                 'Sports Event', 'Author', 'Entertainment', 'Park', 'Person Other', 'Educational Institution',
                 'Businessman', 'Band', 'Character', 'News Agency', 'Location Other', 'TV Works',
                 'Country', 'Artist', 'Festival', 'Company', 'Sports League', 'City', 'Actor', 'Politician', 'Musician',
                 'Sports Team', 'Athlete'],
        'coarse': ['Location', 'Other', 'Art', 'Other', 'Art', 'Product', 'Art', 'Event', 'Organization', 'Location',
                   'Building', 'Person', 'Organization', 'Building', 'Person', 'Other', 'Person', 'Person', 'Art',
                   'Building', 'Product', 'Other', 'Product', 'Other', 'Organization', 'Location', 'Product',
                   'Organization', 'Event', 'Person', 'Building', 'Location', 'Person', 'Organization', 'Person',
                   'Organization', 'Person', 'Organization', 'Location', 'Art', 'Location', 'Person', 'Event',
                   'Organization', 'Organization', 'Location', 'Person', 'Person', 'Person', 'Organization', 'Person'],
        'value': [34, 37, 37, 39, 49, 56, 60, 66, 68, 70, 73, 75, 84, 86, 108, 111, 113, 115, 116, 122, 127, 135, 147, 160,
                  169, 189, 195, 196, 197, 219, 223, 226, 228, 233, 259, 262, 268, 270, 329, 442, 504, 580, 606, 665, 707,
                  873, 934, 1225, 1310, 1614, 1761],
        'groundable_value': [3, 2, 27, 5, 23, 18, 27, 5, 7, 8, 16, 46, 11, 28, 56, 95, 82, 66, 23, 32, 32, 24, 78, 22, 16,
                             7, 98, 29, 35, 68, 30, 27, 129, 37, 161, 137, 169, 55, 57, 58, 30, 198, 119, 162, 98, 12, 745,
                             729, 927, 602, 1299]
    }


    for i in range(len(data['fine'])):
        data['value'][i] -= data['groundable_value'][i]
        
    

    color_discrete_sequence = ['#ee7959', '#fbca4d', '#ffee6f', '#a0d8ef', '#c5e1a5', '#dbb6d6', '#b2ebf2', '#f5f2e9']
    # sunburst_print

    fig2 = make_subplots(rows=1, cols=2, column_width=[1, 0],  # column_width=[0.4,0.6],
                         specs=[[{"type": "domain"}, {"type": "xy"}]]
                         )
    # breakpoint()

    coarse_keys = list(set(data['coarse']))
    labels = data['fine'] + coarse_keys
    parents = [fine2coarse_map[label] if label in fine2coarse_map else '' for label in labels]
    values = data['value'] + [0 for i in labels[len(data['value']):]]

    # colors = [color_discrete_sequence[coarse_keys.index(key)] if key in coarse_keys else color_discrete_sequence[coarse_keys.index(fine2coarse_map[key])] for key in labels]
    # fig2.add_trace(go.Sunburst(
    #     labels=labels,
    #     parents=parents,
    #     values=values,
    #     insidetextorientation='radial',
    #     marker=dict(colors=colors)
    #     ),row=1, col=1)
    # breakpoint()
    fig2.add_trace(go.Bar(
        y=data['fine'][31:],  ## jmwang
        x=data['groundable_value'][31:],  ## jmwang
        orientation='h',
        marker=dict(
            color='#ff85c0',
            # line=dict(color='rgb(248, 248, 249)', width=1)
        ),
        name='Groundable'),
        row=1, col=2
    )
    fig2.add_trace(go.Bar(
        y=data['fine'][31:],  ##jmwamg
        x=data['value'][31:],  ##jmwamg
        orientation='h',
        marker=dict(
            color='#8c8c8c',
            # line=dict(color='rgb(248, 248, 249)', width=1)
        ),
        name='Ungroundable'
    ),
        row=1, col=2
    )
    fig2.update_layout(
        height=480,
        width=540,
        barmode='stack',
        legend=dict(  ##### fig2 legend jmwang
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            title=dict(text=''),
            bordercolor="grey",
            borderwidth=0.5,
            traceorder="reversed",
        ),

        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showline=True,
            linecolor='black',
            mirror=True,
        ),
        yaxis=dict(
            showline=True,
            linecolor='black',
            mirror=True,
            # tickfont=dict(size =8 )
        ),
    )

    fig2.write_image("fig_baronly.pdf", scale=10, format='pdf')


def barcoarse_only():
    ###
    data = {
        'fine': ['Continent', 'Medical Thing', 'Art Other', 'Ordinance', 'Magazine', 'Game', 'Written Work', 'Event Other',
                 'Organization Other', 'Road', 'Cultural Place', 'Director', 'Political Party', 'Sports Facility',
                 'Journalist', 'Animal', 'Coach', 'Intellectual', 'Music', 'Building Other', 'Software', 'Award',
                 'Brand Name Products', 'Website', 'Government Agency', 'State', 'Product Other', 'Social Organization',
                 'Sports Event', 'Author', 'Entertainment', 'Park', 'Person Other', 'Educational Institution',
                 'Businessman', 'Band', 'Character', 'News Agency', 'Location Other', 'TV Works',
                 'Country', 'Artist', 'Festival', 'Company', 'Sports League', 'City', 'Actor', 'Politician', 'Musician',
                 'Sports Team', 'Athlete'],
        'coarse': ['Location', 'Other', 'Art', 'Other', 'Art', 'Product', 'Art', 'Event', 'Organization', 'Location',
                   'Building', 'Person', 'Organization', 'Building', 'Person', 'Other', 'Person', 'Person', 'Art',
                   'Building', 'Product', 'Other', 'Product', 'Other', 'Organization', 'Location', 'Product',
                   'Organization', 'Event', 'Person', 'Building', 'Location', 'Person', 'Organization', 'Person',
                   'Organization', 'Person', 'Organization', 'Location', 'Art', 'Location', 'Person', 'Event',
                   'Organization', 'Organization', 'Location', 'Person', 'Person', 'Person', 'Organization', 'Person'],
        'value': [34, 37, 37, 39, 49, 56, 60, 66, 68, 70, 73, 75, 84, 86, 108, 111, 113, 115, 116, 122, 127, 135, 147, 160,
                  169, 189, 195, 196, 197, 219, 223, 226, 228, 233, 259, 262, 268, 270, 329, 442, 504, 580, 606, 665, 707,
                  873, 934, 1225, 1310, 1614, 1761],
        'groundable_value': [3, 2, 27, 5, 23, 18, 27, 5, 7, 8, 16, 46, 11, 28, 56, 95, 82, 66, 23, 32, 32, 24, 78, 22, 16,
                             7, 98, 29, 35, 68, 30, 27, 129, 37, 161, 137, 169, 55, 57, 58, 30, 198, 119, 162, 98, 12, 745,
                             729, 927, 602, 1299]
    }


    
    for i in range(len(data['fine'])):
        data['value'][i] -= data['groundable_value'][i]

    ## coarse groundable / ungroundable
    coarse_list = ['Person', 'Organization', 'Location', 'Event', 'Art', 'Product', 'Building', 'Other']
    coarse_value = {'Person': 0, 'Organization': 0, 'Location': 0, 'Event': 0, 'Art': 0, 'Product': 0, 'Building': 0,
                    'Other': 0}
    coarse_groundable_value = {'Person': 0, 'Organization': 0, 'Location': 0, 'Event': 0, 'Art': 0, 'Product': 0,
                               'Building': 0, 'Other': 0}
    for i in range(52):
        coarse = data['coarse'][i]
        coarse_value[coarse] += data['value'][i]
        coarse_groundable_value[coarse] += data['groundable_value'][i]
    coarse_value_list = [coarse_value[c] for c in coarse_list[::-1]]
    coarse_groundable_value_list = [coarse_groundable_value[c] for c in coarse_list[::-1]]
    print(coarse_value_list)
    print(coarse_groundable_value_list)

    color_discrete_sequence = ['#ee7959', '#fbca4d', '#ffee6f', '#a0d8ef', '#c5e1a5', '#dbb6d6', '#b2ebf2', '#f5f2e9']
    color_discrete_sequence_light = ['#f3a18b', '#fcda83', '#fff39b', '#bde4f4', '#d7eac0', '#c9f1f6', '#e6cce2',
                                     '#f8f6f0']
    # sunburst_print

    fig2 = make_subplots(rows=1, cols=2, column_width=[1, 0],  # column_width=[0.4,0.6],
                         specs=[[{"type": "domain"}, {"type": "xy"}]]
                         )
    # breakpoint()

    coarse_keys = list(set(data['coarse']))
    labels = data['fine'] + coarse_keys
    parents = [fine2coarse_map[label] if label in fine2coarse_map else '' for label in labels]
    values = data['value'] + [0 for i in labels[len(data['value']):]]

    # colors = [color_discrete_sequence[coarse_keys.index(key)] if key in coarse_keys else color_discrete_sequence[coarse_keys.index(fine2coarse_map[key])] for key in labels]
    # fig2.add_trace(go.Sunburst(
    #     labels=labels,
    #     parents=parents,
    #     values=values,
    #     insidetextorientation='radial',
    #     marker=dict(colors=colors)
    #     ),row=1, col=1)

    fig2.add_trace(go.Bar(
        y=coarse_list[::-1],  ## jmwang
        x=coarse_groundable_value_list,  ## jmwang
        orientation='h',
        marker=dict(
            color='#ff85c0'  # color_discrete_sequence[::-1],#'#ff85c0',
            # line=dict(color='rgb(248, 248, 249)', width=1)
        ),
        name='Groundable'),
        row=1, col=2
    )
    fig2.add_trace(go.Bar(
        y=coarse_list[::-1],  ##jmwamg
        x=coarse_value_list,  ##jmwamg
        orientation='h',
        marker=dict(
            color='#8c8c8c',  # color_discrete_sequence_light[::-1],
            # line=dict(color='rgb(248, 248, 249)', width=1)
        ),
        name='Ungroundable'
    ),
        row=1, col=2
    )
    fig2.update_layout(
        height=480,
        width=540,
        barmode='stack',
        legend=dict(  ##### fig2 legend jmwang
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            title=dict(text=''),
            bordercolor="grey",
            borderwidth=0.5,
            traceorder="reversed",
        ),

        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showline=True,
            linecolor='black',
            mirror=True,
        ),
        yaxis=dict(
            showline=True,
            linecolor='black',
            mirror=True,
            tickfont=dict(size=14)
        ),
    )

    fig2.write_image("fig_barcoarseonly.pdf", scale=10, format='pdf')


# sunburst_print()

sunburst_only()
print('sunburst_only')
bar_only()
print('bar_only')
# barcoarse_only()
