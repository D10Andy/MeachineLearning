# from __future__ import print_function       # 旧版本通过__future__模块的某些功能，测试新版本的新功能
# import sys, os, pdb
# sys.path.insert(0, 'E:\Pythoncode\DeepLearningFighting\StyleTransfer\StyleTransferFollow\src')
# import numpy as np, scipy.misc
# from StyleTransfer.StyleTransferFollow.src.optimize import optimize
# from argparse import ArgumentParser     # 命令行解析
# from StyleTransfer.StyleTransferFollow.src.utils import save_img, get_img, exists,list_files
# import StyleTransfer.StyleTransferFollow.evaluate as evaluate
from __future__ import print_function
import sys, os, pdb
sys.path.insert(0, 'src')
import numpy as np, scipy.misc
from optimize import optimize
from argparse import ArgumentParser
from utils import save_img, get_img, exists, list_files
import evaluate

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2

# 为了使模型不至于过拟合，所以一般将学习率设置的比较小，epoch不需要太大
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
CHECKPOINT_DIR = 'checkpoint'
CHECKPOINT_ITERATIONS = 2000
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = 'data/train2014'
BATCH_SIZE = 4
DEVICE = 'CPU:0'


# 命令行解析
def build_parse():
    parser = ArgumentParser()
    # 在命令行中指定的参数
    parser.add_argument('--checkpoint-dir',type=str, dest='checkpoint_dir',
                        help='dir to save checkpoint',
                        metavar='CHECKPOINT_DIR', required=True)

    parser.add_argument('--style', type=str, dest='style', help='style image path',
                        metavar='STYLE', required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--test', type=str,
                        dest='test', help='test image path',
                        metavar='TEST', default=False)

    parser.add_argument('--test-dir', type=str,
                        dest='test_dir', help='test image save dir',
                        metavar='TEST_DIR', default=False)

    parser.add_argument('--slow', dest='slow', action='store_true',
                        help='gatys\' approach (for debugging, not supported)',
                        default=False)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS',
                        default=CHECKPOINT_ITERATIONS)

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)

    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    return parser


def check_opts(opts):
    # exists() 判断文件路径是否存在
    exists(opts.checkpoint_dir, "checkpoint dir not found!")
    exists(opts.style, " style path not found!")
    exists(opts.train_path, "train path not found!")
    if opts.test or opts.test_dir:
        exists(opts.test, "test img not found!")
        exists(opts.test_dir, "test directory not found!")
    exists(opts.vgg_path, "vgg network data not found!")
    # assert 当这个关键字后面的条件为假时，程序自动崩溃并抛出AssertionError的异常
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert os.path.exists(opts.vgg_path)
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0


def _get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]


def main():
    parser = build_parse()
    # 实际的解析一下这些参数
    options = parser.parse_args()
    check_opts(options)

    style_target = get_img(options.style)
    if not options.slow:
        content_targets = _get_files(options.train_path)
    elif options.test:
        content_targets = [options.test]

    # 指定实际传递的参数
    kwargs = {
        "slow": options.slow,
        "epochs": options.epochs,
        "print_iterations": options.checkpoint_iterations,
        "batch_size": options.batch_size,
        "save_path": os.path.join(options.checkpoint_dir, 'fns.ckpt'),
        "learning_rate": options.learning_rate
    }
    if options.slow:
        if options.epochs < 10:
            kwargs['epochs'] = 1000
        if options.learning_rate < 1:
            kwargs['learning_rate'] = 1e1

    args = [
        content_targets,
        style_target,
        options.content_weight,
        options.style_weight,
        options.tv_weight,
        options.vgg_path
    ]

    for preds, losses, i, epoch in optimize(*args, **kwargs):
        style_loss, content_loss, tv_loss, loss = losses

        print('Epoch %d, Iteration: %d, Loss: %s' % (epoch, i, loss))
        to_print = (style_loss, content_loss, tv_loss)
        print('style: %s, content:%s, tv: %s' % to_print)
        if options.test:
            assert options.test_dir != False
            preds_path = '%s/%s_%s.png' % (options.test_dir, epoch, i)
            if not options.slow:
                ckpt_dir = os.path.dirname(options.checkpoint_dir)
                evaluate.ffwd_to_img(options.test, preds_path,
                                     options.checkpoint_dir)
            else:
                pass
                # save_img(preds_path, img)
    ckpt_dir = options.checkpoint_dir
    cmd_text = 'python evaluate.py --checkpoint %s ...' % ckpt_dir
    print("Training complete. For evaluation:\n    `%s`" % cmd_text)




if __name__ == '__main__':
    main()
