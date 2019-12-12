from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os.path as osp

import cv2
import time
import numpy as np
from PIL import Image
import scipy.io as sio
from data.config import cfg
from models.Res50_RFP import build_model
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr


parser = argparse.ArgumentParser(description='s3fd evaluatuon wider')
parser.add_argument('--model', type=str,
                    default='weights/Res50_RFP.pth', help='trained model')
parser.add_argument('--thresh', default=0.01, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()

#use_cuda=None
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def detect_face(net, img, shrink):
    if shrink != 1:
        img = cv2.resize(img, None, None, fx=shrink, fy=shrink,
                         interpolation=cv2.INTER_LINEAR)
    x = to_chw_bgr(img)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :] 
    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()
    #print('size of original image:',x.size())
    with torch.no_grad():
          y = net(x)    
    detections = y.data
    detections = detections.cpu().numpy()

    det_conf = detections[0, 1, :, 0]
    det_xmin = img.shape[1] * detections[0, 1, :, 1] / shrink
    det_ymin = img.shape[0] * detections[0, 1, :, 2] / shrink
    det_xmax = img.shape[1] * detections[0, 1, :, 3] / shrink
    det_ymax = img.shape[0] * detections[0, 1, :, 4] / shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= args.thresh)[0]
    det = det[keep_index, :]

    return det


def flip_test(net, image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(net, image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def im_det_pyramid(net, image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    det_s = np.row_stack((detect_face(net, image, 0.5), flip_test(net, image, 0.5)))
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 64)[0]
    det_s = det_s[index, :]

    det_temp = np.row_stack((detect_face(net, image, 0.75), flip_test(net, image, 0.75)))
    index = np.where(np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) > 96)[0]
    det_temp = det_temp[index, :]
    det_s = np.row_stack((det_s, det_temp))

    det_temp = np.row_stack((detect_face(net, image, 0.25), flip_test(net, image, 0.25)))
    index = np.where(np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) > 128)[0]
    det_temp = det_temp[index, :]
    det_s = np.row_stack((det_s, det_temp))

    #det_temp = np.row_stack((detect_face(net, image, 0.15), flip_test(net, image, 0.15)))
    #index = np.where(np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) > 32)[0]
    #det_temp = det_temp[index, :]
    #det_s = np.row_stack((det_s, det_temp))    

    st = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
    for i in range(len(st)):
        if (st[i] <= max_im_shrink):
            det_temp = np.row_stack((detect_face(net, image, st[i]), flip_test(net, image, st[i])))
            # Enlarged images are only used to detect small faces.
            if st[i] == 1.25:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 128)[0]
                det_temp = det_temp[index, :]
            elif st[i] == 1.5:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 64)[0]
                det_temp = det_temp[index, :]
            elif st[i] == 1.75:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 32)[0]
                det_temp = det_temp[index, :]
            elif st[i] == 2.0:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 20)[0]
                det_temp = det_temp[index, :]
            elif st[i] == 2.25:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 16)[0]
                det_temp = det_temp[index, :]
            elif st[i] == 2.5:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 14)[0]
                det_temp = det_temp[index, :]
            elif st[i] == 3.0:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 10)[0]
                det_temp = det_temp[index, :]
            det_s = np.row_stack((det_s, det_temp))
    return det_s

def bbox_vote(det):
    if 0==len(det): return[]
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.4)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        #print (merge_index)

        #if merge_index.shape[0] <= 1:
        #    continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(
            det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        #if merge_index.shape[0]<=1:
        #    det_accu_sum=det_accu
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    
    dets = dets[0:750, :]     

    return dets

def get_data():
    subset = 'val'
    if subset is 'val':
        wider_face = sio.loadmat(
            './eval_tools/wider_face_val.mat')
    else:
        wider_face = sio.loadmat(
            './eval_tools/wider_face_test.mat')
    event_list = wider_face['event_list']
    file_list = wider_face['file_list']
    del wider_face

    imgs_path = os.path.join(
        cfg.FACE.WIDER_DIR, 'WIDER_{}'.format(subset), 'images')
    save_path = 'eval_tools/Res50_RFP_{}'.format(subset)

    return event_list, file_list, imgs_path, save_path

if __name__ == '__main__':
    event_list, file_list, imgs_path, save_path = get_data()
    cfg.USE_NMS = False
    net = build_model('test', cfg.NUM_CLASSES)
    print (net)
    net.load_state_dict(torch.load(args.model))
    net.eval()

    if use_cuda:
        net.cuda()
        cudnn.benckmark = True


    #transform = S3FDBasicTransform(cfg.INPUT_SIZE, cfg.MEANS)

    counter = 0
    t_start=time.time()
    for index, event in enumerate(event_list):
        #if index==27:
            filelist = file_list[index][0]
            path = os.path.join(save_path, event[0][0])
            if not os.path.exists(path):
                os.makedirs(path)

            for num, file in enumerate(filelist):
                im_name = file[0][0]
                in_file = os.path.join(imgs_path, event[0][0], im_name[:] + '.jpg')
                #print (in_file)
                #img = cv2.imread(in_file)
                img = Image.open(in_file)
                if img.mode == 'L':
                    img = img.convert('RGB')
                img = np.array(img)

                max_im_shrink = (0x7fffffff / 200.0 /
                                 (img.shape[0] * img.shape[1])) ** 0.5
                print(max_im_shrink)
                #max_im_shrink = np.sqrt(
                #    1700 * 1200 / (img.shape[0] * img.shape[1]))

                shrink = max_im_shrink if max_im_shrink < 1 else 1
                counter += 1

                t1 = time.time()
                det0 = detect_face(net, img, shrink)
                det1 = flip_test(net, img, shrink)    # flip test
                det2 = im_det_pyramid(net, img, max_im_shrink)

                det = np.row_stack((det0, det1, det2))
                #print(det.shape)
                dets = bbox_vote(det)
                #dets=det0

                t2 = time.time()
                print('Detect %04d th image costs %.4f' % (counter, t2 - t1))

                fout = open(osp.join(save_path, event[0][
                            0], im_name + '.txt'), 'w')
                fout.write('{:s}\n'.format(event[0][0] + '/' + im_name + '.jpg'))
                fout.write('{:d}\n'.format(len(dets)))
                if len(dets)>0:
                    for i in range(dets.shape[0]):
                        xmin = dets[i][0]
                        ymin = dets[i][1]
                        xmax = dets[i][2]
                        ymax = dets[i][3]
                        score = dets[i][4]
                        fout.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                                   format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))
    t_end=time.time()
    print('total inference time:', (t_end-t_start))
