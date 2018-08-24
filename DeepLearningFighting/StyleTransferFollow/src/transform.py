import tensorflow as tf, pdb

WEIGHTS_INIT_STDEV = .1


def net(image):
    # 卷积的对象，filter numbers,filter size, stride
    conv1 = _conv_layer(image, 32, 9, 1)
    conv2 = _conv_layer(conv1, 64, 3, 2)
    conv3 = _conv_layer(conv2, 128, 3, 2)
    resid1 = _residual_block(conv3, 3)
    resid2 = _residual_block(resid1, 3)
    resid3 = _residual_block(resid2, 3)
    resid4 = _residual_block(resid3, 3)
    resid5 = _residual_block(resid4, 3)
    conv_t1 = _conv_transpose_layer(resid5, 64, 3, 2)
    conv_t2 = _conv_transpose_layer(conv_t1, 32, 3, 2)
    # 为什么最后一层不用反卷积层
    # 因为原始的图像是三通道的，所以希望最后输出的图像也是三通道的，所以filter的大小是3，size是9
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    # 将值映射到0-255的区间上，由于conv_t3 值的范围是(0,1)
    # ,tanh在（0,1）对应的输出约为（0,0.8），所以*150+255/2 是合适的
    preds = tf.nn.tanh(conv_t3) * 150 + 255. / 2
    return preds


# 卷积层实现
def _conv_layer(net, filter_nums, filter_size, strides, relu=True):
    weights_init = _conv_init_vars(net, filter_nums, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    # 要进行normalization
    net = _instance_norm(net)
    if relu:
        net = tf.nn.relu(net)

    return net


# 反卷积实现
def _conv_transpose_layer(net, filter_nums, filter_size, strides):
    weights_init = _conv_init_vars(net, filter_nums, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)
    # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])
    # 上面就完成了一个上采样的过程
    new_shape = [batch_size, new_rows, new_cols, filter_nums]
    # 将新的shape连接起来
    tf_shape = tf.stack(new_shape)
    strides_shape = [1, strides, strides, 1]

    # 反卷积
    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
    net = _instance_norm(net)
    return tf.nn.relu(net)


# 默认的情况下filter_size=3，所有的残差网络都是定义filter_size=3的
def _residual_block(net, filter_size=3):
    # 一部分是先跑两层卷积，一部分是直接传入的，第二层卷积是不需要relu激活函数的
    tmp = _conv_layer(net, 128, filter_size, 1)
    return net + _conv_layer(tmp, 128, filter_size, 1, relu=False)


# batch normalization
def _instance_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    # 使用tf.nn.moments函数可以直接返回所求特征图的均值和方差,[]中表示的是所求维度
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
    # 下面的也可以使用tf.nn.batch_normalization代替
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    # 根据论文提供的公式
    normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)
    return scale * normalized + shift


# 权重初始化
def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        # 对于反卷积来说，就是in_channels 和out_channels相互调换
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

        weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1),
                                   dtype=tf.float32)
        return weights_init





