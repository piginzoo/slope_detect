if [ "$1" = "" ]; then
    echo "Usage: imgen.sh  <dir:data/raw> <type:train|test|validate>"
    echo "dir:原始图像所在"
    exit
fi

python data_generator/rotate.py --dir $1 --type $2