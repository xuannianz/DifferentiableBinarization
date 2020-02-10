import cv2
import imgaug.augmenters as iaa
import math
import numpy as np
import os.path as osp
import pyclipper
from shapely.geometry import Polygon

from transform import transform, crop, resize

mean = [103.939, 116.779, 123.68]


def load_all_anns(gt_paths, dataset='total_text'):
    res = []
    for gt in gt_paths:
        lines = []
        reader = open(gt, 'r').readlines()
        for line in reader:
            item = {}
            parts = line.strip().split(',')
            label = parts[-1]
            if label == '1':
                label = '###'
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
            if 'icdar' == dataset:
                poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()
            else:
                num_points = math.floor((len(line) - 1) / 2) * 2
                poly = np.array(list(map(float, line[:num_points]))).reshape((-1, 2)).tolist()
            if len(poly) < 3:
                continue
            item['poly'] = poly
            item['text'] = label
            lines.append(item)
        res.append(lines)
    return res


def show_polys(image, anns, window_name):
    for ann in anns:
        poly = np.array(ann['poly']).astype(np.int32)
        cv2.drawContours(image, np.expand_dims(poly, axis=0), -1, (0, 255, 0), 2)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)


def draw_thresh_map(polygon, canvas, mask, shrink_ratio=0.4):
    polygon = np.array(polygon)
    assert polygon.ndim == 2
    assert polygon.shape[1] == 2

    polygon_shape = Polygon(polygon)
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    padded_polygon = np.array(padding.Execute(distance)[0])
    cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

    xmin = padded_polygon[:, 0].min()
    xmax = padded_polygon[:, 0].max()
    ymin = padded_polygon[:, 1].min()
    ymax = padded_polygon[:, 1].max()
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    polygon[:, 0] = polygon[:, 0] - xmin
    polygon[:, 1] = polygon[:, 1] - ymin

    xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
    ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

    distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
    for i in range(polygon.shape[0]):
        j = (i + 1) % polygon.shape[0]
        absolute_distance = compute_distance(xs, ys, polygon[i], polygon[j])
        distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
    distance_map = np.min(distance_map, axis=0)

    xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
    xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
    ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
    ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
    canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
        1 - distance_map[
            ymin_valid - ymin:ymax_valid - ymin,
            xmin_valid - xmin:xmax_valid - xmin],
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])


def compute_distance(xs, ys, point_1, point_2):
    square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) / \
            (2 * np.sqrt(square_distance_1 * square_distance_2))
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)
    result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)

    result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
    return result


def generate(data_dir, batch_size=16, image_size=640, min_text_size=8, shrink_ratio=0.4, thresh_min=0.3,
             thresh_max=0.7, is_training=True):
    split = 'train' if is_training else 'test'
    with open(osp.join(data_dir, f'{split}_list.txt')) as f:
        image_fnames = f.readlines()
        image_paths = [osp.join(data_dir, f'{split}_images', image_fname.strip()) for image_fname in image_fnames]
        gt_paths = [osp.join(data_dir, f'{split}_gts', image_fname.strip() + '.txt') for image_fname in image_fnames]
        all_anns = load_all_anns(gt_paths)
    transform_aug = iaa.Sequential([iaa.Fliplr(0.5), iaa.Affine(rotate=(-10, 10)), iaa.Resize((0.5, 3.0))])
    dataset_size = len(image_paths)
    indices = np.arange(dataset_size)
    if is_training:
        np.random.shuffle(indices)
    current_idx = 0
    b = 0
    while True:
        if current_idx >= dataset_size:
            if is_training:
                np.random.shuffle(indices)
            current_idx = 0
        if b == 0:
            # Init batch arrays
            batch_images = np.zeros([batch_size, image_size, image_size, 3], dtype=np.float32)
            batch_gts = np.zeros([batch_size, image_size, image_size], dtype=np.float32)
            batch_masks = np.zeros([batch_size, image_size, image_size], dtype=np.float32)
            batch_thresh_maps = np.zeros([batch_size, image_size, image_size], dtype=np.float32)
            batch_thresh_masks = np.zeros([batch_size, image_size, image_size], dtype=np.float32)
            batch_loss = np.zeros([batch_size, ], dtype=np.float32)
        i = indices[current_idx]
        image_path = image_paths[i]
        anns = all_anns[i]
        image = cv2.imread(image_path)
        # show_polys(image.copy(), anns, 'before_aug')
        if is_training:
            transform_aug = transform_aug.to_deterministic()
            image, anns = transform(transform_aug, image, anns)
            image, anns = crop(image, anns)
        image, anns = resize(image_size, image, anns)
        # show_polys(image.copy(), anns, 'after_aug')
        # cv2.waitKey(0)
        anns = [ann for ann in anns if Polygon(ann['poly']).is_valid]
        gt = np.zeros((image_size, image_size), dtype=np.float32)
        mask = np.ones((image_size, image_size), dtype=np.float32)
        thresh_map = np.zeros((image_size, image_size), dtype=np.float32)
        thresh_mask = np.zeros((image_size, image_size), dtype=np.float32)
        for ann in anns:
            poly = np.array(ann['poly'])
            height = max(poly[:, 1]) - min(poly[:, 1])
            width = max(poly[:, 0]) - min(poly[:, 0])
            polygon = Polygon(poly)
            # generate gt and mask
            if polygon.area < 1 or min(height, width) < min_text_size or ann['text'] == '###':
                cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                continue
            else:
                distance = polygon.area * (1 - np.power(shrink_ratio, 2)) / polygon.length
                subject = [tuple(l) for l in ann['poly']]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if len(shrinked) == 0:
                    cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                else:
                    shrinked = np.array(shrinked[0]).reshape(-1, 2)
                    if shrinked.shape[0] > 2 and Polygon(shrinked).is_valid:
                        cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
                    else:
                        cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                        continue
            # generate thresh map and thresh mask
            draw_thresh_map(ann['poly'], thresh_map, thresh_mask, shrink_ratio=shrink_ratio)
        thresh_map = thresh_map * (thresh_max - thresh_min) + thresh_min

        image = image.astype(np.float32)
        image[..., 0] -= mean[0]
        image[..., 1] -= mean[1]
        image[..., 2] -= mean[2]
        batch_images[b] = image
        batch_gts[b] = gt
        batch_masks[b] = mask
        batch_thresh_maps[b] = thresh_map
        batch_thresh_masks[b] = thresh_mask

        b += 1
        current_idx += 1
        if b == batch_size:
            inputs = [batch_images, batch_gts, batch_masks, batch_thresh_maps, batch_thresh_masks]
            outputs = batch_loss
            yield inputs, outputs
            b = 0
