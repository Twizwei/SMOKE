import time
import fire
# import kitti_common as kitti
from .eval import get_official_eval_result, get_coco_eval_result
import os
from tqdm import tqdm
import pathlib, re
import numpy as np

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

def get_image_index_str(img_idx):
    return "{:06d}".format(img_idx)

def get_label_annos(label_folder, image_ids=None):
    if image_ids is None:
        filepaths = pathlib.Path(label_folder).glob('*.txt')
        prog = re.compile(r'^\d{6}.txt$')
        filepaths = filter(lambda f: prog.match(f.name), filepaths)
        image_ids = [int(p.stem) for p in filepaths]
        image_ids = sorted(image_ids)
    # print(len(image_ids))
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    annos = []
    label_folder = pathlib.Path(label_folder)
    for idx in tqdm(image_ids):
        # print('{}/{}'.format(i, len(image_ids)))
        image_idx = get_image_index_str(idx)
        label_filename = label_folder / (image_idx + '.txt')
        annos.append(get_label_anno(label_filename))
    return annos
    
def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    # print('starting reading lines...')
    content = [line.strip().split(' ') for line in lines]
    # print('lines read!')
    annotations['name'] = np.array([x[0] for x in content])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]).reshape(
            -1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros([len(annotations['bbox'])])
    return annotations

def evaluate(label_path,
             result_path,
             label_split_file,
             output_dir=None,
             current_class=0,
             difficultys=[0, 1, 2],
             coco=False,
             score_thresh=-1):
    print('loading pred results...')
    dt_annos = get_label_annos(result_path)
    print('pred results loaded!')
    # if score_thresh > 0:
    #     dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    print('loading gt annotations...')
    gt_annos = get_label_annos(label_path, val_image_ids)
    # gt_annos = kitti.get_label_annos(label_path)
    print('gt annotations loaded!')
    if coco:
        print(get_coco_eval_result(gt_annos, dt_annos, current_class))
    else:
        results = get_official_eval_result(gt_annos, dt_annos, current_class, difficultys=difficultys)
        if output_dir is not None:
            f = open(os.path.join(output_dir + 'eval_result.txt'), 'w')
            f.write(results)
            f.close()
        print(results)
        return results


if __name__ == '__main__':
    fire.Fire()
