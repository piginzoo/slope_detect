import multiprocessing
import threading
import time
import logging
import numpy as np
import cv2
import tensorflow as tf
import traceback

logger = logging.getLogger("GeneratorEnqueuer")

try:
    import queue
except ImportError:
    import Queue as queue


class GeneratorEnqueuer():
    def __init__(self, generator,
                 use_multiprocessing=False,
                 wait_time=0.05,
                 random_seed=None):
        self.wait_time = wait_time
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.random_seed = random_seed

    def start(self, workers=1, max_queue_size=10):
        def data_generator_task(name):

            while not self._stop_event.is_set():
                try:
                    if self._use_multiprocessing or self.queue.qsize() < max_queue_size:
                        generator_output = next(self._generator)
                        logger.debug("调用next()，%s 拿到了一批图片，放入queue，当前队列大小( %s / %s )",name,self.queue.qsize(),max_queue_size)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(self.wait_time)
                except BaseException as e:

                    logger.error("加载图片出现异常：", str(e))
                    traceback.format_exc()
                    self._stop_event.set()
                    raise
            logger.info("读取图片进程退出")

        try:
            if self._use_multiprocessing:
                self.queue = multiprocessing.Queue(maxsize=max_queue_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for i in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.random_seed)
                    thread = multiprocessing.Process(target=data_generator_task,
                                                     args=("进程_"+str(i),))
                    thread.daemon = True
                    if self.random_seed is not None:
                        self.random_seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except Exception as e:
            logger.error("加载图片出现异常：", str(e))
            traceback.format_exc()

            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._use_multiprocessing:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._use_multiprocessing:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None

    def get(self):
        while self.is_running():
            if not self.queue.empty():
                inputs = self.queue.get()
                if inputs is not None:
                    yield inputs
            else:
                time.sleep(self.wait_time)

# 看哪个大了，就缩放哪个，规定最大的宽和高：max_width,max_height
def resize_image_list(image_list,max_width,max_height):

    result = []
    for image in image_list:
        h,w,_ = image.shape # H,W

        if h<max_height and w<max_width:
            logger.debug("图片的宽高[%d,%d]比最大要求[%d,%d]小，无需resize",h,w,max_height,max_width)
            result.append(image)

        h_scale = max_height/h
        w_scale = max_width/w
        # print("h_scale",h_scale,"w_scale",w_scale)
        scale = min(h_scale,w_scale) # scale肯定是小于1的，越小说明缩放要厉害，所以谁更小，取谁

        # https://www.jianshu.com/p/11879a49d1a0 关于resize
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        logger.debug("图片从[%d,%d]被resize成为%r",h,w,image.shape)

        result.append(image)
    return result


# 必须按照vgg的要求resize成224x224的，变形就变形了，无所了，另外还要normalize，就是减去那三个值
def prepare4vgg(image_list):

    result = []
    for image in image_list:
        image = cv2.resize(image, (224,224),interpolation=cv2.INTER_AREA)
        image = image[:,:,::-1] # BGR->RGB
        result.append(mean_image_subtraction(image)) #减去均值
    return np.array(result)


# [123.68, 116.78, 103.94] 这个是VGG的预处理要求的，必须减去这个均值：https://blog.csdn.net/smilejiasmile/article/details/80807050
def mean_image_subtraction(images,means=[124, 117, 104]): #means=[123.68, 116.78, 103.94]):
    # 干啥呢？ 按通道，多分出一个维度么？
    for i in range(3):
        images[:,:,i] = images[:,:,i] - means[i]
    return images


if __name__=="__main__":
    data = np.empty([5,7,3])
    data.fill(200)
    data = mean_image_subtraction(data)
    print(data.shape)
    print(data[:,:,0])
    print(data[:, :, 1])
    print(data[:, :, 2])