Date=$(date +%Y%m%d%H%M)

if [ "$1" = "stop" ]; then
    echo "停止倾斜探测服务器"
    ps aux|grep python|grep slope_detect_server|awk '{print $2}'|xargs kill -9
    exit
fi

name=slope_detect \
model_dir=model \
model_file=rotate-2020-05-15-11-54-22-16101.ckpt \
CUDA_VISIBLE_DEVICES=0 \
nohup \
    gunicorn \
    web.slope_detect_server:app \
    --workers=1 \
    --worker-class=gevent \
    --bind=0.0.0.0:8080 \
    --timeout=300 \
    >> ./logs/slope_detect_$Date.log 2>&1 &