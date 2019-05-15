if [ "$1" = "" ]; then
    echo "select.sh <图片目录> <备份目录>"
    exit
fi

echo "开始检测图片的倾斜....."

CUDA_VISIBLE_DEVICES=0 python main/select.py \
    --pred_dir=data/pred \
    --debug=True \
    --target=$2 \
    --model_dir=model \
    --model_file=ctpn-2019-05-07-14-19-35-201.ckpt