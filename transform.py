import imgaug
import cv2
import numpy as np


def transform(aug, image, anns):
    image_shape = image.shape
    image = aug.augment_image(image)
    new_anns = []
    for ann in anns:
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in ann['poly']]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints, shape=image_shape)])[0].keypoints
        poly = [(min(max(0, p.x), image.shape[1] - 1), min(max(0, p.y), image.shape[0] - 1)) for p in keypoints]
        new_ann = {'poly': poly, 'text': ann['text']}
        new_anns.append(new_ann)
    return image, new_anns


def split_regions(axis):
    regions = []
    min_axis_index = 0
    for i in range(1, axis.shape[0]):
        if axis[i] != axis[i - 1] + 1:
            region = axis[min_axis_index:i]
            min_axis_index = i
            regions.append(region)
    return regions


def random_select(axis):
    xx = np.random.choice(axis, size=2)
    xmin = np.min(xx)
    xmax = np.max(xx)
    return xmin, xmax


def region_wise_random_select(regions):
    selected_index = list(np.random.choice(len(regions), 2))
    selected_values = []
    for index in selected_index:
        axis = regions[index]
        xx = int(np.random.choice(axis, size=1))
        selected_values.append(xx)
    xmin = min(selected_values)
    xmax = max(selected_values)
    return xmin, xmax


def crop(image, anns, max_tries=10, min_crop_side_ratio=0.1):
    h, w, _ = image.shape
    h_array = np.zeros(h, dtype=np.int32)
    w_array = np.zeros(w, dtype=np.int32)
    for ann in anns:
        points = np.round(ann['poly'], decimals=0).astype(np.int32)
        minx = np.min(points[:, 0])
        maxx = np.max(points[:, 0])
        w_array[minx:maxx] = 1
        miny = np.min(points[:, 1])
        maxy = np.max(points[:, 1])
        h_array[miny:maxy] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]

    if len(h_axis) == 0 or len(w_axis) == 0:
        return image, anns

    h_regions = split_regions(h_axis)
    w_regions = split_regions(w_axis)

    for i in range(max_tries):
        if len(w_regions) > 1:
            xmin, xmax = region_wise_random_select(w_regions)
        else:
            xmin, xmax = random_select(w_axis)
        if len(h_regions) > 1:
            ymin, ymax = region_wise_random_select(h_regions)
        else:
            ymin, ymax = random_select(h_axis)

        if xmax - xmin < min_crop_side_ratio * w or ymax - ymin < min_crop_side_ratio * h:
            # area too small
            continue
        new_anns = []
        for ann in anns:
            poly = np.array(ann['poly'])
            if not (poly[:, 0].min() > xmax
                    or poly[:, 0].max() < xmin
                    or poly[:, 1].min() > ymax
                    or poly[:, 1].max() < ymin):
                poly[:, 0] -= xmin
                poly[:, 0] = np.clip(poly[:, 0], 0., (xmax - xmin - 1) * 1.)
                poly[:, 1] -= ymin
                poly[:, 1] = np.clip(poly[:, 1], 0., (ymax - ymin - 1) * 1.)
                new_ann = {'poly': poly.tolist(), 'text': ann['text']}
                new_anns.append(new_ann)

        if len(new_anns) > 0:
            return image[ymin:ymax, xmin:xmax], new_anns

    return image, anns


def resize(size, image, anns):
    h, w, c = image.shape
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    h = int(h * scale)
    w = int(w * scale)
    padimg = np.zeros((size, size, c), image.dtype)
    padimg[:h, :w] = cv2.resize(image, (w, h))
    new_anns = []
    for ann in anns:
        poly = np.array(ann['poly']).astype(np.float64)
        poly *= scale
        new_ann = {'poly': poly.tolist(), 'text': ann['text']}
        new_anns.append(new_ann)
    return padimg, new_anns
