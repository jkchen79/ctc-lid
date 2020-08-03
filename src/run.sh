
# set -x

mode=$1

[ -z $mode ] && mode='train'

train_utt2npy='../data/train_utt2npy'
train_utt2lang='../data/train_utt2lang'
train_utt2phones_seq='../data/train_utt2phones_seq'

dev_utt2npy='../data/dev_utt2npy'
dev_utt2lang='../data/dev_utt2lang'
dev_utt2phones_seq='../data/dev_utt2phones_seq'

eval_utt2npy='../data/dev_utt2npy'
eval_utt2lang='../data/dev_utt2lang'
eval_utt2phones_seq='../data/dev_utt2phones_seq'

languages='../data/lang.list.txt'
phones_list='../data/phones.list.txt'

dt=`date +%m%d`

if [ $mode == 'train' ]
then
    gpus='4,5,6,7'
    job_name="resnet18_bilstmL1_jointLearning"
    ckpt_dir="../ckpt/${job_name}_$dt"
    log_file="./log/job_train_ctc_lid_${job_name}_${dt}.log.txt"

    CUDA_VISIBLE_DEVICES=$gpus python ctc_main.py \
        --do-train \
        --train-utt2npy $train_utt2npy  \
        --train-utt2target $train_utt2lang \
        --train-utt2label-seq $train_utt2phones_seq \
        --eval-utt2npy $eval_utt2npy \
        --eval-utt2target $eval_utt2lang \
        --eval-utt2label-seq $eval_utt2phones_seq \
        --targets-list $languages \
        --labels-list $phones_list \
        --ckpt-dir $ckpt_dir \
        --log-file $log_file \
        --pretrain-ctc-epochs 60 \
        --pretrain-ctc-lr 0.001 \
        --epochs 100 \
        --lr 0.001 \
        --padding-batch \
        --bidirectional \
        --batch-size 64 \
        --hidden-size 512 \
        --num-rnn-layers 1


elif [ $mode == 'test' ]
then
    ckpt_path='../ckpt/ckpt-best_model/final.mdl'
    gpus='0,1,2,3,4,5,6,7'

    testset_name="ch_lid_dialect"
    log_file="./log/job_test_ctc_lid_${dt}-${testset_name}.log.txt"

    CUDA_VISIBLE_DEVICES=$gpus python ctc_main.py \
        --do-eval \
        --eval-utt2npy $eval_utt2npy \
        --eval-utt2target $eval_utt2lang \
        --eval-utt2label-seq $eval_utt2phones_seq \
        --targets-list $languages \
        --labels-list $phones_list \
        --ckpt $ckpt_path \
        --log-file $log_file
        --padding-batch \
        --batch-size 64 \
        --bidirectional \
        --hidden-size 1024 \
        --num-rnn-layers 1 


fi
