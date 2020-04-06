# coding=utf-8
import logging
from PIL import Image, ImageDraw, ImageFont, ImageFilter,ImageOps
import random
import os
logger = logging.getLogger("rotate")
ROTATE_ANGLE = 5        # 随机旋转角度
NUM_IN_DIRECTION = 2     # 每个方向上的数量
classes_name = ["正图","90度","180度","270度"]
degrees = [0,90,180,270]


def init_logger():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()])


def rotate_in_background(img,angle):
    global bg_list
    back_img = random.choice(bg_list)
    img = img.convert('RGBA')
    img = img.rotate(angle,expand = 1)
    # fff = Image.new('RGBA',img.size,(255,)*4)
    # img = Image.composite(img,fff,img)
    # img = img.crop((0, 0, img.size[0], img.size[1]))
    back_img = back_img.crop((0, 0, img.size[0], img.size[1]))
    back_img.paste(img,(0,0),img)
    return back_img


def degree_rotate(img,degree):
    # 宽和高
    w, h = img.size
    # 中心点
    center = (w // 2, h // 2)
    logger.debug("指定角度旋转度数:%f" % degree)
    return img.rotate(degree,center=center,expand=1)


# 生成一张图片, 1200x1920
def load_all_backgroud_images(bground_path):
    bground_list = []

    for img_name in os.listdir(bground_path):
        image = Image.open( os.path.join(bground_path , img_name))
        if image.mode == "L":
            logger.error("图像[%s]是灰度的，转RGB",img_name)
            image = image.convert("RGB")

        bground_list.append(image)
        logger.debug("加载背景图片：%s",bground_path + img_name)
    logger.debug("所有图片加载完毕")

    return bground_list



def process_folder(dir,type,num):
    label_file_name = os.path.join("data",type+".txt")

    label_file = open(label_file_name,"w")
    target_dir = os.path.join("data",type)

    count = 0
    for image_name in os.listdir(dir):
        if count> num: break

        file_name = os.path.join(dir, image_name)
        _,type = os.path.splitext(file_name)
        if type in ['.jpg','.png','.JPG','.jpeg','.PNG']:
            logger.debug("处理原始图片:%s", file_name)
            rotate_one_image(file_name,label_file,target_dir)
            count += 1
        else:
            logger.warning("警告：文件%s不是图片，忽略",file_name)
            continue

    label_file.close()


#旋转一张图片
def rotate_one_image(raw_image_name,label_file,target_dir):
    for i in range(4):
        d = degrees[i]
        # 原图
        image = Image.open(raw_image_name)
        # 按4个方向要求旋转先
        image = degree_rotate(image,d)
        # 得到文件名和后缀
        (filepath, tempfilename) = os.path.split(raw_image_name)
        name,subfix = os.path.splitext(tempfilename)
        name+= "_" + str(i)
        logger.debug("产生%s的图：%s", classes_name[i], name)
        rotate_one_direction(image,name,subfix,label_file,target_dir,i)


# 旋转一个角度，每一个角度生成5~10个样本并存储
def rotate_one_direction(image,image_name,subfix,label_file,target_dir,clazz):
    random_num = NUM_IN_DIRECTION #random.randint(5,10)  # 5~10的随机数

    # 生成旋转角度，最后需要一张旋转角度为0的
    degrees = []
    for i in range(random_num):
        small_degree = random.uniform(-ROTATE_ANGLE, ROTATE_ANGLE)  # 随机旋转0-30度
        degrees.append(small_degree)
    degrees.append(0)

    logger.debug("%s随机旋转角度：%r",image_name,degrees)

    i = 0
    for degree in degrees:
        name = image_name+"_"+str(i)
        rotated_file_name = os.path.join(target_dir,name+subfix)
        # rotated_file = degree_rotate(image,degree)
        rotated_file = rotate_in_background(image,degree)
        try:
            rotated_file.save(rotated_file_name)
        except IOError as e:
            logger.error("保存图片失败：%s,原因：%s",rotated_file_name,str(e))
            continue

        label_file.write(rotated_file_name)
        label_file.write(" ")
        label_file.write(str(clazz))
        label_file.write("\n")

        logger.debug("保存旋转后的图片&标签：%s", rotated_file_name)
        i+= 1


# 把指定目录下的文件都旋转，生成测试和验证代码
if __name__ == '__main__':

    import argparse

    init_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--type")       # 啥类型的数据啊，train/validate/test
    parser.add_argument("--dir")        # 这个程序的主目录
    parser.add_argument("--background") # 背景图的目录
    parser.add_argument("--num")

    args = parser.parse_args()
    DATA_DIR = args.dir
    TYPE= args.type
    NUM = int(args.num)

    images_dir = os.path.join("data",TYPE)
    if not os.path.exists(images_dir): os.makedirs(images_dir)

    global bg_list
    bg_list = load_all_backgroud_images(args.background)


    process_folder(DATA_DIR, TYPE,NUM)
