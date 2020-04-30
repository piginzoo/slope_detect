if [ "$1" = "help" ]; then
    echo "pred.sh 使用说明：
    --image_dir     被预测的图片目录
    --pred_dir      预测后的结果的输出目录
    --model_dir     model的存放目录，会自动加载最新的那个模型
    --model_file    model的模型文件 "
    exit
fi


if [ "$1" = "old" ]; then
    echo "老模型开始检测图片的倾斜....."
    python main/pred.py \
        --gpu=1 \
        --image_name=$1 \
        --pred_dir=data/validate \
        --debug=True \
        --model_dir=model \
        --model_file=ctpn-2019-05-07-14-19-35-201.ckpt
    exit
fi


echo "新模型开始检测图片的倾斜....."
nohup python main/pred.py \
    --gpu=0 \
    --image_name=$1 \
    --pred_dir=data/validate \
    --debug=True \
    --model_dir=model \
    --model_file=rotate-2020-04-29-22-34-44-5001.ckpt \
    >> ./logs/pred_rotate_gpu0_$Date.log 2>&1 &