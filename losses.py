import tensorflow as tf
import keras.backend as K


def balanced_crossentropy_loss(args, negative_ratio=3., scale=5.):
    pred, gt, mask = args
    pred = pred[..., 0]
    positive_mask = (gt * mask)
    negative_mask = ((1 - gt) * mask)
    positive_count = tf.reduce_sum(positive_mask)
    negative_count = tf.reduce_min([tf.reduce_sum(negative_mask), positive_count * negative_ratio])
    loss = K.binary_crossentropy(gt, pred)
    positive_loss = loss * positive_mask
    negative_loss = loss * negative_mask
    negative_loss, _ = tf.nn.top_k(tf.reshape(negative_loss, (-1,)), tf.cast(negative_count, tf.int32))

    balanced_loss = (tf.reduce_sum(positive_loss) + tf.reduce_sum(negative_loss)) / (
            positive_count + negative_count + 1e-6)
    balanced_loss = balanced_loss * scale
    return balanced_loss, loss


def dice_loss(args):
    """

    Args:
        pred: (b, h, w, 1)
        gt: (b, h, w)
        mask: (b, h, w)
        weights: (b, h, w)
    Returns:

    """
    pred, gt, mask, weights = args
    pred = pred[..., 0]
    weights = (weights - tf.reduce_min(weights)) / (tf.reduce_max(weights) - tf.reduce_min(weights)) + 1.
    mask = mask * weights
    intersection = tf.reduce_sum(pred * gt * mask)
    union = tf.reduce_sum(pred * mask) + tf.reduce_sum(gt * mask) + 1e-6
    loss = 1 - 2.0 * intersection / union
    return loss


def l1_loss(args, scale=10.):
    pred, gt, mask = args
    pred = pred[..., 0]
    mask_sum = tf.reduce_sum(mask)
    loss = K.switch(mask_sum > 0, tf.reduce_sum(tf.abs(pred - gt) * mask) / mask_sum, tf.constant(0.))
    loss = loss * scale
    return loss


def db_loss(args):
    binary, thresh_binary, gt, mask, thresh, thresh_map, thresh_mask = args
    l1_loss_ = l1_loss([thresh, thresh_map, thresh_mask])
    balanced_ce_loss_, dice_loss_weights = balanced_crossentropy_loss([binary, gt, mask])
    dice_loss_ = dice_loss([thresh_binary, gt, mask, dice_loss_weights])
    return l1_loss_ + balanced_ce_loss_ + dice_loss_
