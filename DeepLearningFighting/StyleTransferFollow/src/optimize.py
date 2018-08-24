# import StyleTransfer.StyleTransferFollow.src.vgg as vgg
# import pdb, time, os
# import functools
# import numpy as np
# from StyleTransfer.StyleTransferFollow.src.utils import get_img
# import tensorflow as tf
# import StyleTransfer.StyleTransferFollow.src.transform as transform

from __future__ import print_function
import functools
import vgg, pdb, time
import tensorflow as tf, numpy as np, os
import transform
from utils import get_img

# 定义特征图
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'


def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=False):
    if slow:
        batch_size = 1
    # 如果数据的个数不是batch_size的整数倍，就将最后面多余的数据舍弃
    mod = len(content_targets) % batch_size
    if mod > 0:
        content_targets = content_targets[:-mod]

    # style feature 要在多个特征图上比较
    style_features = []
    batch_shape = (batch_size, 256, 256, 3)
    style_shape = (1,) + style_target.shape

    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        # 指定一系列的placeholder
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        # style image 的预处理操作
        style_image_pre = vgg.preprocess(style_image)
        # vgg的net
        net = vgg.net(vgg_path, style_image_pre)
        style_pre = np.array(style_target)
        for layer in STYLE_LAYERS:
            # 取出当前层的features
            features = net[layer].eval(feed_dict={style_image: style_pre})
            # 我们要对比的并不是每一个特征图上数值之间的差异，
            # 而是比较的特征图之间的差异值，所以关心的整体的特征值
            # -1表示以特征图为单位，shape的第三维表示的是特征图
            features = np.reshape(features, (-1, features.shape[3]))  # 没有理解是如何取出特征图整体的
            gram = np.matmul(features.T, features)      # 特征图和特征图之间的关系
            style_features[layer] = gram

    with tf.Graph.as_default(), tf.Session() as sess:
        x_content = tf.placeholder(tf.float32, shape=batch_shape, name='x-content')
        x_pre = vgg.preprocess(x_content)
        content_features = []
        content_net = vgg.net(vgg_path, x_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        # 3CNN, 5残差网络, 2CNNT
        preds = transform.net(x_content/255.0)

        # 特征图之间的差异值
        preds_pre = vgg.preprocess(preds)
        net = vgg.net(vgg_path, preds_pre)
        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        # content loss
        content_loss = content_weight*(2*tf.nn.l2_loss(
            net[CONTENT_LAYER]-content_features[CONTENT_LAYER])/content_size)

        # style loss
        style_loss = []
        for style_layer in STYLE_LAYERS:
            # 在生成网络上生成的图像，在特征图上的风格上差异是多少
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i: i.value, layer.get_shape())
            size = height*width*filters
            feats = tf.reshape(layer, (bs, height*width, filters))
            feats_t = tf.transpose(feats, perm=[0, 2, 1])
            grams = tf.matmul(feats_t, feats)/size

            # 不用生成网络的差异值是多少
            style_gram = style_features[style_layer]
            style_loss.append(2*tf.nn.l2_loss(grams-style_gram)/style_gram.size)

        style_losses = style_weight*functools.reduce(tf.add, style_loss)/batch_size

        # total variation denoising(降噪)
        tv_y_size = _tensor_size(preds[:, 1:, :, :])
        tv_x_size = _tensor_size(preds[:, :, 1:, :])
        y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :batch_shape[1] - 1, :, :])
        x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :batch_shape[2] - 1, :])
        tv_loss = tv_weight * 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size

        loss = content_loss + style_losses + tv_loss
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())

        import random
        uid = random.randint(1, 100)
        print("UID: %s" % uid)
        for epoch in range(epochs):
            num_examples = len(content_targets)
            iterations = 0
            while iterations*batch_size < num_examples:
                start_time = time.time()
                curr = iterations*batch_size
                step = curr + batch_size
                x_batch = np.zeros(shape=batch_shape, dtype=tf.float32)
                for j,img_p in enumerate(content_targets[curr:step]):
                    x_batch[j] = get_img(img_p, (256, 256, 3)).astype(tf.float32)
                iterations +=1
                assert x_batch.shape[0] == batch_size

                feed_dict = {x_content: x_batch}

                train_step.run(feed_dict=feed_dict)

                end_time = time.time()
                delta_time = end_time-start_time
                if debug:
                    print("UID: %s, batch time: %s" % (uid, delta_time))
                is_print_iter = int(iterations) % print_iterations == 0
                if slow:
                    is_print_iter = epoch % print_iterations == 0
                is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
                should_print = is_print_iter or is_last
                if should_print:
                    to_get = [style_loss, content_loss, tv_loss, loss, preds]
                    test_dict = {x_content: x_batch}
                    tup = sess.run(to_get, feed_dict=test_dict)
                    _style_loss, _content_loss, _tv_loss, _loss, _preds = tup
                    losses = (_style_loss, _content_loss, _tv_loss, _loss)
                    # 每隔一段时间进行保存模型
                    if slow:
                       _preds = vgg.unprocess(_preds)
                    else:
                       saver = tf.train.Saver()
                       res = saver.save(sess, save_path)
                    # yield 是一个类似return的关键字，只是这个函数返回的是个生成器，
                    # 当你调用这个函数的时候，函数内部的代码并不立马执行，
                    # 这个函数只是返回一个生成器对象。当你使用for进行迭代时，
                    # 函数中的代码才会执行
                    yield(_preds, losses, iterations, epoch)


def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)




