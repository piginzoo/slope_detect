from utils import data_provider as data_provider
import time,logging
import traceback

def init_logger():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()])
init_logger()

logger = logging.getLogger("test")

data_generator = data_provider.get_batch(num_workers=10, label_file="data/train.txt",batch_num=64)
counter = 0
for step in range(100000000):
    try:
        image_list, label_list = next(data_generator)  # next(<迭代器>）来返回下一个结果
        counter+= len(image_list)
        logger.info(counter)
    except Exception as e:
        traceback.format_exc()
        logger.error("报错啦：%s",str(e))



