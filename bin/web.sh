Date=$(date +%Y%m%d%H%M)

if [ "$1" = "stop" ]; then
    echo "停止训练"
    ps aux|grep python|grep slope_detect|awk '{print $2}'|xargs kill -9
    exit
fi


nohup \
name=slope_detect \
model_dir=model \
model_file=ctpn-2019-05-07-14-19-35-201.ckpt \
CUDA_VISIBLE_DEVICES=0 \
    gunicorn \
    web.server:app \
    --workers=1 \
    --worker-class=gevent \
    --bind=0.0.0.0:8080 \
    --timeout=300 \
    >> ./logs/slope_detect_$Date.log 2>&1 &