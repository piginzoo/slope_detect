# 默认CTPN用GPU1，CRNN用GPU0

Date=$(date +%Y%m%d%H%M)

python -m main.validate \
    --validate_dir=data/validate \
    --validate_batch=1 \
    --validate_times=100 \
    --validate_label=data/validate.txt \
