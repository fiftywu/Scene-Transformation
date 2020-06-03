"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch

from data.base_dataset import BaseDataset, get_params, get_transform
from DarknetTest import Detector
from PIL import Image
import numpy as np



class Loss:
    def __init__(self):
        self.aver_L1loss = 0
        self.aver_inL1loss = 0
        self.aver_outL1loss = 0
        self.TVloss = 0

class GenMask:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.detector = Detector(gpu=opt.gpu_ids[0],
                                 cfg="/home/fiftywu/fiftywu/Files/DeepLearning/pix/pix2pixAgain/darknet/cfg/yolov3.cfg",
                                 weights="/home/fiftywu/fiftywu/Files/DeepLearning/pix/pix2pixAgain/darknet/yolov3.weights",
                                 data="/home/fiftywu/fiftywu/Files/DeepLearning/pix/pix2pixAgain/darknet/cfg/coco.data")


    def get_suitablemask(self, data):
        # data['Raw_A']
        # data['Transform_params']
        transform_params = data['Transform_params']
        tmp = transform_params['crop_pos']
        crop_pos = (tmp[0].item(), tmp[1].item())
        flip = transform_params['flip'].item()
        tp = {'crop_pos': crop_pos, 'flip': flip}

        mask_transform = get_transform(self.opt, tp, grayscale=True)
        Raw_A = data['Raw_A'][0].numpy()  # numpy
        result, widhei = self.detector.detect_result(Raw_A)
        dmask = self.detector.detect_mask(result, widhei, remove_list=['person', 'car', 'truck', 'bus'])  # numpy
        im = Image.fromarray(dmask)
        dmask = mask_transform(im).to(self.device) == 1  # True is Dynamic part
        dmask = dmask.expand(1, 1, 256, 256)
        return dmask

    def get_L1loss(self, visuals, dmask=True, scale='-1-1'):
        mask_num = torch.sum(dmask)+1e-3
        unmask_num = torch.sum(~dmask)+1e-3
        image_num = visuals['fake_B'].shape[1]*visuals['fake_B'].shape[2]*visuals['fake_B'].shape[3]
        overall_L1loss = torch.sum(torch.abs(visuals['real_B']-visuals['fake_B']))/image_num
        mask_L1loss = torch.sum(torch.abs(visuals['real_B']*dmask-visuals['fake_B']*dmask))/mask_num
        unmask_L1loss = torch.sum(torch.abs(visuals['real_B']*(~dmask)-visuals['fake_B']*(~dmask)))/unmask_num
        if scale=='-1-1':
            return overall_L1loss.item(), mask_L1loss.item(), unmask_L1loss.item()
        elif scale=='0-1':
            return overall_L1loss.item()/2, mask_L1loss.item()/2, unmask_L1loss.item()/2



if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    overall_L1loss_save = []
    mask_L1loss_save = []
    unmask_L1loss_save = []
    GenMask = GenMask(opt)
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        # fiftywu
        dmask = GenMask.get_suitablemask(data)
        overall_L1loss, mask_L1loss, unmask_L1loss = GenMask.get_L1loss(visuals, dmask, scale='0-1')
        overall_L1loss_save.append(overall_L1loss)
        mask_L1loss_save.append(mask_L1loss)
        unmask_L1loss_save.append(unmask_L1loss)

        #
        img_path = model.get_image_paths()     # get image paths
        if i % 200 == 0:  # save images to an HTML file
            # print('processing (%04d)-th image... %s' % (i, img_path))
            print('##', i,
                  'overall_L1loss=', np.mean(overall_L1loss_save),
                  'mask_L1loss=', np.mean(mask_L1loss_save),
                  'unmask_L1loss=', np.mean(unmask_L1loss_save),
                  'number=', len(dataset))
        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    # webpage.save()  # save the HTML

    print('overall_L1loss=',np.mean(overall_L1loss_save),
          'mask_L1loss=',np.mean(mask_L1loss_save),
          'unmask_L1loss=', np.mean(unmask_L1loss_save),
          'number=', len(dataset))
    print('pause')