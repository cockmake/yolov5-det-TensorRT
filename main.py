import threading
import time

import tensorrt as trt
import torch
from collections import OrderedDict, namedtuple
import base64
import cv2 as cv
import numpy as np
import os


# 模型部分
confidence = 0.5
nms_threshold = 0.4
input_shape = (1, 3, 640, 640)
N, C, H, W = input_shape
logger = trt.Logger(trt.Logger.INFO)
device = torch.device('cuda:0')
Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
bindings, bindings2 = OrderedDict(), OrderedDict()
bindings_addrs, bindings_addrs2 = OrderedDict(), OrderedDict()
context, context2 = None, None


def format_img(np_img):
    # YOLOV5 输入的预处理
    _H, _W, _ = np_img.shape
    im = np.zeros((640, 640, 3), dtype=np.uint8)
    im[...] = 114
    factor_w = _W / 640
    factor_h = _H / 640
    factor = max(factor_w, factor_h)
    img = cv.resize(np_img, (int(_W / factor), int(_H / factor)))
    _H, _W, _ = img.shape
    dif_w = int((640 - _W) / 2)
    dif_h = int((640 - _H) / 2)
    im[dif_h: dif_h + _H, dif_w: dif_w + _W] = img

    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).float().to(device)  # float() 之关键  float() 之关键  float() 之关键  float() 之关键
    im /= 255  # 0 - 255 to 0.0 - 1.0
    # 返回一些预处理后的图像, 和预处理时附加的信息
    return im, dif_w, dif_h, factor

# 加载模型
def load_model_2d():
    global bindings_addrs, context
    with open('2D_16.engine', 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = model.get_binding_shape(index)
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        bindings_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        context = model.create_execution_context()

def load_model_3d():
    global bindings_addrs2, context2
    with open('3D_16.engine', 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = model.get_binding_shape(index)
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings2[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        bindings_addrs2 = OrderedDict((n, d.ptr) for n, d in bindings2.items())
        context2 = model.create_execution_context()


def process_output(img_src, output_2d):
    src_h, src_w, _ = img_src.shape
    indices = cv.dnn.NMSBoxes(output_2d[:, :4], output_2d[:, 4], confidence, nms_threshold)
    boxes = output_2d[indices, :4]
    class_ids = output_2d[indices, 5]
    return boxes, class_ids


def draw_on_src(img_src, boxes, class_ids):
    for box in boxes:
        box = box.astype(np.int32)
        cv.rectangle(img_src, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 6)

out_prob_2D, out_prob_3D = None, None
def model_process_2D(img_src_2D):
    # 获取处理的2D结果
    global out_prob_2D
    img_blob, dif_w, dif_h, factor = format_img(img_src_2D)
    bindings_addrs['images'] = img_blob.data_ptr()
    context.execute_v2(list(bindings_addrs.values()))
    out_prob = bindings['output0'].data.squeeze()
    # 以下这段在GPU上操作, 并且要用矩阵操作, 可以节约时间, 处理成NMSBOXES函数可接受的数据格式
    out_prob = out_prob[out_prob[:, 4] > confidence]
    out_prob[:, 0] -= out_prob[:, 2] / 2 + dif_w
    out_prob[:, 1] -= out_prob[:, 3] / 2 + dif_h
    out_prob[:, :4] *= factor
    value, idx = torch.max(out_prob[:, 5:] * out_prob[:, 4:5], dim=1)  # 相乘拿到真正概率
    out_prob[:, 4] = value  # 写入confidence
    out_prob[:, 5] = idx
    out_prob = out_prob[:, :6].cpu().numpy()  # 处理完后把数据转到CPU上做最后的NMS处理
    out_prob_2D = out_prob

def model_process_3D(img_src_3D):  # 获取处理的3D结果
    # 3D图像做相同的处理
    global out_prob_3D
    img_blob, dif_w, dif_h, factor = format_img(img_src_3D)
    bindings_addrs2['images'] = img_blob.data_ptr()
    context2.execute_v2(list(bindings_addrs2.values()))
    out_prob = bindings2['output0'].data.squeeze()
    out_prob = out_prob[out_prob[:, 4] > confidence]
    out_prob[:, 0] -= out_prob[:, 2] / 2 + dif_w
    out_prob[:, 1] -= out_prob[:, 3] / 2 + dif_h
    out_prob[:, :4] *= factor
    value, idx = torch.max(out_prob[:, 5:] * out_prob[:, 4:5], dim=1)
    out_prob[:, 4] = value  # 写入confidence
    out_prob[:, 5] = idx
    out_prob = out_prob[:, :6].cpu().numpy()
    out_prob_3D = out_prob


def process():
    img_2D_path = "Orange2D_L783.jpg"
    img_3D_path = "Orange3D_L783.jpg"
    img_src_2D = cv.imread(img_2D_path)
    img_src_3D = cv.imread(img_3D_path)
    t = time.time()
    # 由于存在线程处理返回结果结果的接收问题, 把out_prob_2D, out_prob_3D结果设置为全局变量
    th1 = threading.Thread(target=model_process_2D, args=(img_src_2D, ))
    th2 = threading.Thread(target=model_process_3D, args=(img_src_3D, ))
    th1.start(), th2.start()  # 启动两个线程
    th1.join(), th2.join()  # 等待两个线程都完成

    out_prob = np.concatenate([out_prob_2D, out_prob_3D], axis=0)
    boxes, class_ids = process_output(img_src_2D, out_prob)
    draw_on_src(img_src_2D, boxes, class_ids)
    cv.imwrite("ret.jpg", img_src_2D)
    # threading.Thread(target=img_save, args=(img_src_2D,)).start()  # 如果要保存图像的话可以开线程 节省处理时间
    fps = 1 / (time.time() - t)
    print(fps)

def main():
    load_model_2d()
    load_model_3d()
    process()

if __name__ == '__main__':
    main()



