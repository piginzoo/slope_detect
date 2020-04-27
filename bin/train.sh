# 默认CTPN用GPU1，CRNN用GPU0

Date=$(date +%Y%m%d%H%M)

if [ "$1" = "stop" ]; then
    echo "停止训练"
    ps aux|grep python|grep rotate_detect|awk '{print $2}'|xargs kill -9
    exit
fi

if [ "$1" = "console" ]; then
    echo "调试模式:只训练一次"
    python -m main.train \
        --name=rotate_detect \
        --pretrained_model_path=data/vgg_16.ckpt \
        --num_readers=1 \
        --max_steps=1 \
        --decay_steps=1 \
        --evaluate_steps=1 \
        --validate_dir=data/check0427/validate \
        --validate_batch=1 \
        --validate_times=1 \
        --validate_label=data/check0427/validate.txt \
        --train_dir=data/train \
        --train_batch=2 \
        --train_label=data/check0427/train.txt \
        --learning_rate=0.01 \
        --save_checkpoint_steps=2000 \
        --decay_rate=0.1 \
        --lambda1=1000 \
        --gpu=1\
        --debug=True \
        --logs_path=logs \
        --moving_average_decay=0.997 \
        --restore=False \
        --early_stop=5 \
        --max_width=1200 \
        --max_height=1600
    exit
fi

echo "生产模式:GPU1"
nohup python -m main.train \
    --name=rotate_detect \
    --pretrained_model_path=data/vgg_16.ckpt \
    --num_readers=8 \
    --max_steps=100000 \
    --decay_steps=1000 \
    --evaluate_steps=100 \
    --validate_dir=data/validate \
    --validate_batch=1 \
    --validate_times=32 \
    --validate_label=data/validate.txt \
    --train_dir=data/train \
    --train_batch=3 \
    --train_label=data/train.txt \
    --learning_rate=0.0001 \
    --save_checkpoint_steps=5000 \
    --decay_rate=0.5 \
    --lambda1=1000 \
    --gpu=1 \
    --debug=False \
    --logs_path=logs \
    --moving_average_decay=0.997 \
    --restore=False \
    --early_stop=100 \
    --max_width=600 \
    --max_height=800 \
    >> ./logs/rotate_gpu0_$Date.log 2>&1 &
