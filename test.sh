python3 validate.py --arch=CLIP:ViT-L/14 --ckpt=checkpoints/clip_vitl14/model_epoch_14.pth --result_folder=results/clip_vitl14_14ep < /dev/null > val_clip_vitl14_14ep.txt 2>&1 &
wait; python3 validate.py --arch=CLIP:ViT-L/14 --ckpt=checkpoints/clip_vitl14_pretrained/model_epoch_16.pth --result_folder=results/clip_vitl14_16ep_pretrained < /dev/null > val_clip_vitl14_16ep_pretrained.txt 2>&1 &
wait; python3 validate.py --arch=CLIP:RN50 --ckpt=checkpoints/clip_rn50/model_epoch_40.pth --result_folder=results/clip_rn50_40ep < /dev/null > val_clip_rn50_40ep.txt 2>&1 &
