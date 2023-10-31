import os


def _bio_tag_to_spans(tags, ignore_labels=None):
    r"""
    给定一个tags的list，比如['O', 'B-singer', 'I-singer', 'I-singer', 'O', 'O']。
        返回[('singer', (1, 4))] (左闭右开区间)

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()
    
    spans = []
    prev_bio_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == 'b':
            spans.append((label, [idx, idx]))
        elif bio_tag == 'i' and prev_bio_tag in ('b', 'i') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bio_tag == 'o':  # o tag does not count
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bio_tag = bio_tag
    return [(span[0], (span[1][0], span[1][1] + 1)) for span in spans if span[0] not in ignore_labels]


input_dir = 'Twitter10000v2/txt_fine'
output_dir = './'

for ss in ['train.txt', 'dev.txt', 'test.txt']:
    f = open(os.path.join(input_dir, ss), 'r')

    new_lines = []

    sentence = []
    label = []
    img_id = ''
    for line in f:
        if line.startswith('IMGID:'):
            img_id = line.strip()[6:]
        elif line == '\n':
            ## Despite the cold, the crowd soon warmed to support act, The Rising Souls.####[['The Rising Souls', 'Other', 'NULL', 'NULL']]####2255a63c0010cbd9d9cf35788417d6ed.jpg
            tags = _bio_tag_to_spans(label)
            tuples = []
            for tag, (i, j) in tags:
                tuples.append([' '.join(sentence[i:j]), tag, 'NULL', 'NULL'])
            new_line = ' '.join(sentence) + '####' + str(tuples) + '####' + img_id + '.jpg'
            new_lines.append(new_line)

            sentence = []
            label = []
            img_id = ''
        else:
            line = line.strip().split('\t')
            sentence.append(line[0])
            label.append(line[-1])

    tags = _bio_tag_to_spans(label)
    tuples = []
    for tag, (i, j) in tags:
        tuples.append(' '.join(sentence[i:j], tag, 'NULL', 'NULL'))
    new_line = ' '.join(sentence) + '####' + str(tuples) + '####' + img_id + '.jpg'
    new_lines.append(new_line)
    with open(os.path.join(output_dir, ss), 'w') as fw:
        for line in new_lines:
            fw.write(line + '\n')
