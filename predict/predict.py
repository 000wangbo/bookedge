import argparse
import os

import paddle

from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, logger, get_image_list
from paddleseg.core import predict
from paddleseg.transforms import Compose
import sys

def get_test_config(cfg):

    test_config = cfg.test_config
#     test_config['aug_pred'] = True
#     test_config['scales'] = [0.5, 1.0, 1.5, 2.0]
#     test_config['flip_horizontal'] = True
#     if 'aug_eval' in test_config:
#         test_config.pop('aug_eval')
#     if args.aug_pred:
#         test_config['aug_pred'] = args.aug_pred
#         test_config['scales'] = args.scales

#     if args.flip_horizontal:
#         test_config['flip_horizontal'] = args.flip_horizontal

#     if args.flip_vertical:
#         test_config['flip_vertical'] = args.flip_vertical

#     if args.is_slide:
#         test_config['is_slide'] = args.is_slide
#         test_config['crop_size'] = args.crop_size
#         test_config['stride'] = args.stride


    test_config['custom_color'] = [0, 0, 0, 255, 255, 255]

    return test_config


def main():
    env_info = get_sys_env()
    assert len(sys.argv) == 3
    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]

    place = 'gpu'


    paddle.set_device(place)

    cfg = Config('bisenet_bookedge_512x512.yml')

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model
    transforms = Compose(cfg.val_transforms)
    image_list, image_dir = get_image_list(src_image_dir)
    logger.info('Number of predict images = {}'.format(len(image_list)))

    test_config = get_test_config(cfg)

    predict(
        model,
        model_path='model/model.pdparams',
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=save_dir,
        **test_config)


if __name__ == '__main__':
    main()
