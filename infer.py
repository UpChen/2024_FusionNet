import numpy as np
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import joint_transforms
from config import VMD_test_root
from misc import check_mkdir
# from networks.TVSD import TVSD
from networks.FusionNet import FusionNet
from dataset.VShadow_crosspairwise_query_other import listdirs_only, CrossPairwiseImg
import argparse
from tqdm import tqdm
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = {
    'scale': 512,
    'test_adjacent': 1,
    'input_folder': 'JPEGImages',
    'label_folder': 'Annotations'
}

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

val_joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args['scale'], args['scale']))
])

root = VMD_test_root[0]

to_pil = transforms.ToPILImage()

val_set = CrossPairwiseImg([VMD_test_root], val_joint_transform, img_transform, target_transform)
val_loader = DataLoader(val_set, batch_size=1, num_workers=1, shuffle=False)


def main():
    net = FusionNet().cuda()

    
    checkpoint = ''
    save_dir = ""
    check_point = torch.load(checkpoint)
    net.load_state_dict(check_point['model'])

    net.eval()
    with torch.no_grad():
        old_temp = ''
        val_iterator = tqdm(val_loader)
        for i, sample in enumerate(val_iterator):
            exemplar, query, other = sample['exemplar'].cuda(), sample['query'].cuda(), sample['other'].cuda()
            # exemplar_gt, query_gt, other_gt = sample['exemplar_gt'].cuda(), sample['query_gt'].cuda(), sample['other_gt'].cuda()
            # exemplar_gt, query_gt = sample['exemplar_gt'].cuda(), sample['query_gt'].cuda()
            video_name = sample['video_name'][0]

            if old_temp == video_name:
                # exemplar_index = query_index
                query_index = query_index + 1
            else:
                # query_index = 1
                query_index = 0
                # exemplar_index = 1
                # #
                # if flag is False:
                #     cv2.imwrite(os.path.join(save_dir, "exemplar_results", str(old_temp), query_save_name),
                #                 exemplar_prediction)
                #     flag = True
                # else:
                #     Image.fromarray(exemplar_prediction).save(
                #         os.path.join(seg_save_dir, "exemplar_results", str(old_temp), query_save_name))

            # print(exemplar_index)
            # print(query_index)
            # print()
            _, _, _, query_final, _ = net(exemplar, query, other)
            res = (query_final.data > 0).to(torch.float32).squeeze(0)
            # exemplar_res = (exemplar_pre.data > 0).to(torch.float32).squeeze(0)
            # res = torch.sigmoid(exemplar_pre.squeeze())

            # query_index1 = str(query_index).zfill(4)
            query_index1 = str(query_index).zfill(5)
            # exemplar_index1 = str(exemplar_index).zfill(4)
            first_image = np.array(Image.open(root + '/JPEGImages/' + str(video_name) + '/' + query_index1 + '.jpg'))
            # print(first_image.shape)
            h, w, _ = first_image.shape

            prediction = np.array(
                transforms.Resize((h, w))(to_pil(res.cpu())))
            # exemplar_prediction = np.array(
            #     transforms.Resize((h, w))(to_pil(exemplar_res.cpu())))

            check_mkdir(os.path.join(save_dir, str(video_name)))
            # check_mkdir(os.path.join(seg_save_dir, "exemplar_results", str(video_name)))

            query_save_name = f"{query_index1}.png"
            # exemplar_save_name = f"{exemplar_index1}.png"
            # print(os.path.join(seg_save_dir, "results", video, save_name))
            Image.fromarray(prediction).save(os.path.join(save_dir, str(video_name), query_save_name))
            # Image.fromarray(exemplar_prediction).save(
            #     os.path.join(seg_save_dir, "exemplar_results", str(video_name), exemplar_save_name))

            old_temp = video_name



def sortImg(img_list):
    img_int_list = [int(f) for f in img_list]
    sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
    return [img_list[i] for i in sort_index]


def getAdjacentIndex(current_index, start_index, video_length, adjacent_length):
    if current_index + adjacent_length < start_index + video_length:
        query_index_list = [current_index+i+1 for i in range(adjacent_length)]
    else:
        query_index_list = [current_index-i-1 for i in range(adjacent_length)]
    return query_index_list

if __name__ == '__main__':
    main()
