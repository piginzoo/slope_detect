#!/usr/bin/env bash
if [ "$1" = "" ]; then
    echo "dir:工程根目录 -> project path  sum:使用主图像个数"
    exit
fi

python data_generator/gen_img_template.py --dir $1 --sum $2