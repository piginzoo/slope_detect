if [ "$1" = "help" ]; then
    echo "pb_pred.sh 使用说明：
    --image_dir     被预测的图片目录
    --pred_dir      预测后的结果的输出目录
    --model_path     model的存放目录，会自动加载最新的那个模型 "
    exit
fi

echo "开始检测图片的倾斜....."

python main/pb_pred.py \
    --gpu=0 \
    --image_name=$1 \
    --pred_dir=data/validate \
    --debug=True \
    --model_path=model/pb/100000