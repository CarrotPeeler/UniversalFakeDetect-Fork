python3 train.py --name=clip_vitl14_pre_1ep --wang2020_data_path=/home/vislab-001/Jared/dip/CS545-Real-Fake-Image-Detection/sentry-dataset/ --data_mode=dip  --arch=CLIP:ViT-L/14  --fix_backbone --save_epoch_freq=1 --gpu_ids=0 --num_threads=4 --batch_size=256 --niter=1 --ckpt=./pretrained_weights/fc_weights.pth < /dev/null > vit-pre-1ep.txt 2>&1 &
wait; python3 train.py --name=clip_vitl14_1ep --wang2020_data_path=/home/vislab-001/Jared/dip/CS545-Real-Fake-Image-Detection/sentry-dataset/ --data_mode=dip  --arch=CLIP:ViT-L/14  --fix_backbone --save_epoch_freq=1 --gpu_ids=0 --num_threads=4 --batch_size=256 --niter=1 < /dev/null > vit-1ep.txt 2>&1 &
wait; python3 validate.py --arch=CLIP:ViT-L/14 --ckpt=checkpoints/clip_vitl14_1ep/model_epoch_0.pth --result_folder=results/clip_vitl14_1ep < /dev/null > val_clip_vitl14_1ep.txt 2>&1 &
wait; python3 validate.py --arch=CLIP:ViT-L/14 --ckpt=checkpoints/clip_vitl14_pre_1ep/model_epoch_0.pth --result_folder=results/clip_vitl14_pre_1ep < /dev/null > val_clip_vitl14_pre_1ep.txt 2>&1 &