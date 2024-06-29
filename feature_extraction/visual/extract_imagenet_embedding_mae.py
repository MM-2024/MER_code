import os
import torch
import torchvision
from torchvision import transforms
import numpy as np
import argparse
import sys

sys.path.append('../../')
import my_config
from dataset import FaceDataset

sys.path.append('/home/mer/MERTools/MER2024/tools/transformers/MAE-Face')
import models_vit

#python -u extract_imagenet_embedding_mae.py   --dataset=MER2024 --feature_level='UTTERANCE'         --gpu=0    


def extract(data_loader, model):
    model.eval()
    with torch.no_grad():
        features, timestamps =  [], []
        [], []
        for images, names in data_loader:
            images = images.cuda()
            embedding = model(images)
            embedding = embedding.squeeze()  # Reduce dimensions from [32, 512, 1, 1] to [32, 512]
            features.append(embedding.cpu().detach().numpy())
            timestamps.extend(names)
        features, timestamps = np.row_stack(features), np.array(timestamps)
        return features, timestamps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--dataset', type=str, default='BoxOfLies', help='input dataset')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', help='feature level [FRAME or UTTERANCE]')
    parser.add_argument('--gpu', type=str, default='1', help='gpu id')
    params = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu

    print('==> Extracting imagenet embedding...')
    face_dir = my_config.PATH_TO_RAW_FACE[params.dataset]
    save_dir = os.path.join(my_config.PATH_TO_FEATURES[params.dataset], f'imagenet_mae_{params.feature_level[:3]}')
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # load model
    model_name = 'vit_base_patch16'
    ckpt_path = '/home/mer/MERTools/MER2024/tools/transformers/MAE-Face/mae_face_pretrain_vit_base.pth'
    model = getattr(models_vit, model_name)(
        global_pool=True,
        num_classes=12,
        drop_path_rate=0.1,
        img_size=224,
    )
    model = model.cuda()

    print(f"Load pre-trained checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_model = checkpoint['model']
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)  # Print which weights are not loaded

    # transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Extract embedding video by video
    vids = os.listdir(face_dir)
    print(f'Find total "{len(vids)}" videos.')
    for i, vid in enumerate(vids, 1):
        print(f"Processing video '{vid}' ({i}/{len(vids)})...")
        video_path = os.path.join(face_dir, vid)
        if not os.path.exists(video_path):
            print(f"Directory not found for video {vid}")
            continue  # 跳过当前循环的其余部分
        dataset = FaceDataset(vid, face_dir, transform=transform)
        if len(dataset) == 0:
            print(f"Warning: number of frames of video {vid} should not be zero.")
            embeddings, framenames = [], []
        else:
            data_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=32,
                                                      num_workers=4,
                                                      pin_memory=True)
            embeddings, framenames = extract(data_loader, model)

        indexes = np.argsort(framenames)
        embeddings = embeddings[indexes]
        framenames = framenames[indexes]
        csv_file = os.path.join(save_dir, f'{vid}.npy')
        np.save(csv_file, embeddings.squeeze())  # Save features
