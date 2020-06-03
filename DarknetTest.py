from darknet import darknet
import numpy as np
import matplotlib.pyplot as plt


def detect_item_mask(item_mask, item):
    # item_mask: True,dynamic; False, static
    # item center_x, center_y, width, height
    left_top = list(map(lambda x: round(x),[item[2][0]-item[2][2]/2., item[2][1]-item[2][3]/2.]))
    right_buttom = list(map(lambda x: round(x),[item[2][0]+item[2][2]/2., item[2][1]+item[2][3]/2.]))
    item_mask[left_top[1]:right_buttom[1], left_top[0]:right_buttom[0]] = True
    return item_mask

class Detector:
    def __init__(self, gpu, cfg, weights, data):
        darknet.set_gpu(gpu)
        self.net = darknet.load_net(str.encode(cfg),
                          str.encode(weights), 0)
        self.meta = darknet.load_meta(str.encode(data))

    def detect_result(self, img):
        try:
            img = str.encode(img)
        except:
            pass
        result, widhei = darknet.detect(self.net, self.meta, img)
        return result, widhei

    def detect_mask(self, result, widhei, remove_list=['person', 'car']):
        # wh, width, height
        remove_list_encode = list(map(lambda x: str.encode(x), remove_list))
        mask = np.zeros([widhei[1], widhei[0]], dtype=np.bool)
        for item in result:
            if item[0] in remove_list_encode:
                item_mask = detect_item_mask(mask, item)
                mask = mask+item_mask
        return mask



if __name__ == '__main__':
    img_path = "/home/fiftywu/fiftywu/Files/DeepLearning/pix/pix2pixAgain/darknet/data/dog.jpg"
    yolov3 = Detector(gpu=1,
                      cfg="/home/fiftywu/fiftywu/Files/DeepLearning/pix/pix2pixAgain/darknet/cfg/yolov3.cfg",
                      weights="/home/fiftywu/fiftywu/Files/DeepLearning/pix/pix2pixAgain/darknet/yolov3.weights",
                      data="/home/fiftywu/fiftywu/Files/DeepLearning/pix/pix2pixAgain/darknet/cfg/coco.data")
    r, s = yolov3.detect_result(img_path)
    mask = yolov3.detect_mask(r, s, remove_list=['dog'])
    plt.imshow(mask)

