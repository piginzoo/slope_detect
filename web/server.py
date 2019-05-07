#-*- coding:utf-8 -*- 
from flask import Flask,jsonify,request,render_template
import base64,cv2,numpy as np,logging
from threading import current_thread
from main import pred
import os
cwd = os.getcwd()
app = Flask(__name__,root_path="web")
app.jinja_env.globals.update(zip=zip)

logger = logging.getLogger("WebServer")

#读入的buffer是个纯byte数据
def process(buffer):
    logger.debug("从web读取数据len:%r",len(buffer))
    if len(buffer)==0: return False,"Image is null"
    data_array = np.frombuffer(buffer,dtype=np.uint8)
    image = cv2.imdecode(data_array, cv2.IMREAD_COLOR)
    if image is None:
        logger.error("图像解析失败")#有可能从字节数组解析成图片失败
        return "图像角度探测失败"

    logger.debug("从字节数组变成图像的shape:%r",image.shape)

    global input_images,sess
    _classes = pred.pred(sess, classes, input_images, np.array([image]))

    return "图片旋转角度为[{}]".format(pred.CLASS_NAME[_classes[0]])


@app.route("/")
def index():
    # with open("../version") as f:
    #     version = f.read()
    return render_template('index.html',version="version")


# base64编码的图片识别
@app.route('/rotate.64',methods=['POST'])
def ocr_base64():

    base64_data = request.form.get('image','')

    # 去掉可能传过来的“data:image/jpeg;base64,”HTML tag头部信息
    index = base64_data.find(",")
    if index!=-1: base64_data = base64_data[index+1:]

    buffer = base64.b64decode(base64_data)
    
    try:
        result = process(buffer)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error("处理图片过程中出现问题：%r",e)
        return jsonify({'result':str(e)})

    return jsonify({'result': result})


# 图片的识别
@app.route('/ocr',methods=['POST'])
def ocr():
    data = request.files['image']
    image_name = data.filename
    buffer = data.read()
    logger.debug("获得上传图片[%s]，尺寸：%d 字节", image_name,len(buffer))

    try:
        result = process(buffer)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error("处理图片过程中出现问题：%r", e)
        result = str(e)

    return render_template('result.html', result=result)


# 图片的识别
@app.route('/test',methods=['GET'])
def test():
    with open("test/test.png", "rb") as f:
        data = base64.b64encode(f.read())
        data = str(data, 'utf-8')
    aa = ['a2','a1']
    bb = ['b2','b1']
    return render_template('test.html', data=data,aa=aa,bb=bb)


def startup():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()])
    logger.debug('子进程:%s,父进程:%s,线程:%r', os.getpid(), os.getppid(), current_thread())
    logger.debug("初始化TF各类参数")
    logger.debug('web目录:%s', app.root_path)

    global input_images,classes,sess

    # gunicorn没法用--方式传参，只好用环境变量了
    model_dir = os.environ['model_dir']
    model_name = os.environ['model_file']
    logger.info("模型目录：%s",model_dir)
    logger.info("模型名称：%s",model_name)

    pred.init_params(model_dir,model_name)
    input_images,classes = pred.init_model()
    sess = pred.restore_session()

    # # 测试代码
    # with open("test/test.png","rb") as f:
    #     image = f.read()
    # process(image,"test.jpg")


startup()