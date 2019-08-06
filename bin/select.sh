if [ "$1" = "" ]; then
    echo "select.sh <图片目录> <备份目录>"
    exit
fi

echo "开始检测图片的倾斜....."

CUDA_VISIBLE_DEVICES=1 python -m main.select.select \
    --pred_dir=$1 \
    --debug=True \
    --target_dir=$2 \
    --model_dir=model \
    --model_file=ctpn-2019-05-07-14-19-35-201.ckpt