if [ "$1" = "" ]; then
    echo "Usage: imgen.sh  <dir:data/raw> <type:train|test|validate> <background dir> <num>"
    echo "dir:原始图像所在 , background:背景所在目录"
    exit
fi

python data_generator/rotate.py --dir $1 --type $2 --background $3 --num $4