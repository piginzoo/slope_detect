#!/usr/bin/env bash
if [ "$1" = "" ]; then
    echo "dir:数据根目录  sum:主图像个数 repeat:重复执行次数，默认1"
    exit
fi

python data_generator/gen_img_template.py --dir $1 --sum $2 --repeat $3