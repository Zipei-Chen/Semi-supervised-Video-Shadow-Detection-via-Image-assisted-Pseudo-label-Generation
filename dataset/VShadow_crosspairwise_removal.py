import os
import os.path

import torch.utils.data as data
from PIL import Image
import random
import torch
import numpy as np
import pdb
import joint_transforms
from torchvision import transforms

validation_list = ['airplane', 'airplane_dance', 'Ball_ce2', 'baseball', 'baseball_girl1', 'baseball_girl2', 'basketball-12',
       'Bicycle', 'bike1', 'bike3', 'Bike_ce1', 'blanket', 'BlurCar2', 'bolt2', 'bottle', 'bottle_move',
       'cap-waving', 'car_drifting', 'car_rc_rotating', 'Couple', 'cup', 'cup2', 'cushion', 'dance1', 'dance2',
       'David3', 'dog1', 'dog2', 'double-walk', 'finger_guessing', 'Fish_ce1', 'food', 'gesture', 'giraffe-19',
       'gorilla-15', 'hand_wave', 'headphone', 'helicopter1', 'Human9', 'jump1', 'jump2', 'kangaroo-13',
       'lizard-18', 'Motorbike_ce', 'person-3', 'person_scooter', 'pig_bag', 'Plane_ce2', 'rabbit', 'roadblock',
       'roadway_car', 'shake-hands', 'sheep-6', 'skateboard-18', 'skateboard-19', 'skateboard-20',
       'skateboard-4', 'Skiing', 'Skiing_ce', 'Skiing_red', 'soccer_ball_2', 'sponge1', 'swing-18',
       'swing-7', 'table', 'Tennis_ce1', 'walk1', 'walk2', 'walk4', 'wave']


# return image triple pairs in video and return single image
class CrossPairwiseImg(data.Dataset):
    def __init__(self, root, joint_transform=None, img_transform=None, target_transform=None):
        self.img_root, self.video_root = self.split_root(root)
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.input_folder = 'images'
        self.label_folder = 'labels'
        self.img_ext = '.jpg'
        self.label_ext = '.png'
        self.num_video_frame = 0
        # get all frames from video datasets
        self.videoImg_list = self.generateImgFromVideo(self.video_root)
        print('Total video frames is {}.'.format(self.num_video_frame))
        # get all frames from image datasets
        if len(self.img_root) > 0:
            self.singleImg_list = self.generateImgFromSingle(self.img_root)
            print('Total single image frames is {}.'.format(len(self.singleImg_list)))


    def __getitem__(self, index):
        manual_random = random.random()  # random for transformation
        # pair in video
        exemplar_path, exemplar_gt_path, videoStartIndex, videoLength = self.videoImg_list[index]  # exemplar
        # sample from same video
        query_index = np.random.randint(videoStartIndex, videoStartIndex + videoLength)
        if query_index == index:
            query_index = np.random.randint(videoStartIndex, videoStartIndex + videoLength)
        query_path, query_gt_path, videoStartIndex2, videoLength2 = self.videoImg_list[query_index]  # query
        if videoStartIndex != videoStartIndex2 or videoLength != videoLength2:
            raise TypeError('Something wrong')
        # sample from different video
        while True:
            other_index = np.random.randint(0, self.__len__())
            if other_index < videoStartIndex or other_index > videoStartIndex + videoLength - 1:
                break  # find image from different video
        other_path, other_gt_path, videoStartIndex3, videoLength3 = self.videoImg_list[other_index]  # other
        if videoStartIndex == videoStartIndex3:
            raise TypeError('Something wrong')
        # single image in image dataset
        if len(self.img_root) > 0:
            single_idx = np.random.randint(0, videoLength)
            single_image_path, single_gt_path = self.singleImg_list[single_idx]  # single image

        # read image and gt
        exemplar = Image.open(exemplar_path).convert('RGB')
        query = Image.open(query_path).convert('RGB')
        other = Image.open(other_path).convert('RGB')
        exemplar_gt = Image.open(exemplar_gt_path).convert('L')
        query_gt = Image.open(query_gt_path).convert('L')
        other_gt = Image.open(other_gt_path).convert('L')
        if len(self.img_root) > 0:
            single_image = Image.open(single_image_path).convert('RGB')
            single_gt = Image.open(single_gt_path).convert('L')

        # transformation
        if self.joint_transform is not None:
            exemplar, exemplar_gt = self.joint_transform(exemplar, exemplar_gt, manual_random)
            query, query_gt = self.joint_transform(query, query_gt, manual_random)
            other, other_gt = self.joint_transform(other, other_gt)
            if len(self.img_root) > 0:
                single_image, single_gt = self.joint_transform(single_image, single_gt)
        if self.img_transform is not None:
            exemplar = self.img_transform(exemplar)
            query = self.img_transform(query)
            other = self.img_transform(other)
            if len(self.img_root) > 0:
                single_image = self.img_transform(single_image)
        if self.target_transform is not None:
            exemplar_gt = self.target_transform(exemplar_gt)
            query_gt = self.target_transform(query_gt)
            other_gt = self.target_transform(other_gt)
            if len(self.img_root) > 0:
                single_gt = self.target_transform(single_gt)
        if len(self.img_root) > 0:
            sample = {'exemplar': exemplar, 'exemplar_gt': exemplar_gt, 'query': query, 'query_gt': query_gt,
                  'other': other, 'other_gt': other_gt, 'single_image': single_image, 'single_gt': single_gt}
        else:
            sample = {'exemplar': exemplar, 'exemplar_gt': exemplar_gt, 'query': query, 'query_gt': query_gt,
                  'other': other, 'other_gt': other_gt}
        return sample

    def generateImgFromVideo(self, root):
        imgs = []
        root = root[0]  # assume that only one video dataset
        video_list = os.listdir(os.path.join(root[0], self.input_folder))
        for video in video_list:
            img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root[0], self.input_folder, video)) if f.endswith(self.img_ext)] # no ext
            img_list = self.sortImg(img_list)
            for img in img_list:
                # videoImgGt: (img, gt, video start index, video length)
                videoImgGt = (os.path.join(root[0], self.input_folder, video, img + self.img_ext),
                        os.path.join(root[0], self.label_folder, video, img + self.label_ext), self.num_video_frame, len(img_list))
                imgs.append(videoImgGt)
            self.num_video_frame += len(img_list)
        return imgs

    def generateImgFromSingle(self, root):
        imgs = []
        for sub_root in root:
            tmp = self.generateImagePair(sub_root[0])
            imgs.extend(tmp)  # deal with image case
            print('Image number of ImageSet {} is {}.'.format(sub_root[2], len(tmp)))

        return imgs

    def generateImagePair(self, root):
        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, self.input_folder)) if f.endswith(self.img_ext)]
        if len(img_list) == 0:
            raise IOError('make sure the dataset path is correct')
        return [(os.path.join(root, self.input_folder, img_name + self.img_ext), os.path.join(root, self.label_folder, img_name + self.label_ext))
            for img_name in img_list]

    def sortImg(self, img_list):
        img_int_list = [int(f) for f in img_list]
        sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
        return [img_list[i] for i in sort_index]

    def split_root(self, root):
        if not isinstance(root, list):
            raise TypeError('root should be a list')
        img_root_list = []
        video_root_list = []
        for tmp in root:
            if tmp[1] == 'image':
                img_root_list.append(tmp)
            elif tmp[1] == 'video':
                video_root_list.append(tmp)
            else:
                raise TypeError('you should input video or image')
        return img_root_list, video_root_list

    def __len__(self):
        return len(self.videoImg_list)//2*2


class pseudo_VSD_Dataset(data.Dataset):
    def __init__(self, root, joint_transform=None, img_transform=None, target_transform=None, is_train = True):
        self.img_root, self.video_root = self.split_root(root)
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.input_folder = 'images'
        self.label_folder = 'labels'
        self.img_ext = '.jpg'
        self.label_ext = '.png'
        self.num_video_frame = 0
        # get all frames from video datasets
        self.videoImg_list = self.generateImgFromVideo(self.video_root, is_train)
        print('Total video frames is {}.'.format(self.num_video_frame))
        # get all frames from image datasets
        if len(self.img_root) > 0:
            self.singleImg_list = self.generateImgFromSingle(self.img_root)
            print('Total single image frames is {}.'.format(len(self.singleImg_list)))

    def __getitem__(self, index):
        manual_random = random.random()  # random for transformation
        # pair in video
        # pdb.set_trace()
        exemplar_path, exemplar_gt_path, uncertainty_path, videoStartIndex, videoLength = self.videoImg_list[index]  # exemplar
        clip_length = 6
        clip_index = [index + i for i in range(clip_length)]
        # ref_index1 = index-1
        # ref_index2 = index+1
        # if ref_index1 < videoStartIndex:
        # if index == videoStartIndex:
        #     ref_index1 = videoStartIndex+1
        #     ref_index2 = videoStartIndex+2
        # if ref_index2 > videoStartIndex+videoLength-1:
        #     ref_index1 = videoStartIndex + videoLength - 2
        #     ref_index2 = videoStartIndex + videoLength - 1
        if index + clip_length > videoStartIndex + videoLength:
            clip_index = [videoStartIndex + videoLength - 6 + i for i in range(clip_length)]
            # ref_index1 = videoStartIndex+videoLength-2
            # ref_index2 = videoStartIndex+videoLength-1

        # ref_index1 = index - 5
        # ref_index2 = index + 5
        # if ref_index1 < videoStartIndex:
        # # if index == videoStartIndex:
        #     ref_index1 = videoStartIndex + 5
        #     ref_index2 = videoStartIndex + 10
        # if ref_index2 > videoStartIndex+videoLength-1:
        #     ref_index1 = videoStartIndex + videoLength - 5
        #     ref_index2 = videoStartIndex + videoLength - 10
        # # elif index == videoStartIndex + videoLength - 1:
        # #     ref_index1 = videoStartIndex + videoLength - 2
        # #     ref_index2 = videoStartIndex + videoLength - 1

        # sample from same video
        clip_info = []
        for i in range(clip_length):
            path, gt_path, uncertaint_path, start_index, length = self.videoImg_list[clip_index[i]]
            clip_info.append([path, gt_path, uncertaint_path, start_index, length])
        # ref1_path, ref1_gt_path, videoStartIndex2, videoLength2 = self.videoImg_list[ref_index1]
        # ref2_path, ref2_gt_path, videoStartIndex3, videoLength3 = self.videoImg_list[ref_index2]

        # print(exemplar_path + ' ' + ref1_path + ' ' + ref2_path)
        for i in range(clip_length):
            if videoStartIndex != clip_info[i][3]:
                raise TypeError('wrong')
        # single image in image dataset

        # read image and gt and uncertainty
        input_clip = []
        gt_clip = []
        uncertainty_clip = []
        resize_clip = []
        path_clip = []
        h_clip = []
        w_clip = []
        for i in range(clip_length):
            path_clip.append(clip_info[i][0])
            exemplar = Image.open(clip_info[i][0]).convert('RGB')
            h_clip.append(exemplar.size[1])
            w_clip.append(exemplar.size[0])
            exemplar_gt = Image.open(clip_info[i][1]).convert('L')
            exemplar_uncertain = Image.open(clip_info[i][2]).convert('L')
            if self.joint_transform is not None:
                exemplar, exemplar_gt, exemplar_uncertain = self.joint_transform(exemplar, exemplar_gt, exemplar_uncertain, manual_random)
                exemplar_resize = transforms.Resize([384, 384])(exemplar)
                exemplar = self.img_transform(exemplar)
                exemplar_resize = transforms.ToTensor()(exemplar_resize)
                exemplar_gt = self.target_transform(exemplar_gt)
                exemplar_uncertain = self.target_transform(exemplar_uncertain)
            input_clip.append(exemplar)
            gt_clip.append(exemplar_gt)
            uncertainty_clip.append(exemplar_uncertain)
            resize_clip.append(exemplar_resize)

        # exemplar = Image.open(clip_info[0][0]).convert('RGB')
        # ref1 = Image.open(ref1_path).convert('RGB')
        # ref2 = Image.open(ref2_path).convert('RGB')
        # exemplar_gt = Image.open(exemplar_gt_path).convert('L')
        # ref1_gt = Image.open(ref1_gt_path).convert('L')
        # ref2_gt = Image.open(ref2_gt_path).convert('L')
        #
        # # print(exemplar.size)
        #
        #
        #
        # # transformation
        # if self.joint_transform is not None:
        #     # resize_gt = transforms.Resize([50, 50])(exemplar_gt)
        #     resize_gt = exemplar_gt
        #     exemplar, exemplar_gt = self.joint_transform(exemplar, exemplar_gt, manual_random)
        #     ref1, ref1_gt = self.joint_transform(ref1, ref1_gt, manual_random)
        #     ref2, ref2_gt = self.joint_transform(ref2, ref2_gt, manual_random)
        #     exemplar_resize = transforms.Resize([384, 384])(exemplar)
        #     ref1_resize = transforms.Resize([384, 384])(ref1)
        #     ref2_resize = transforms.Resize([384, 384])(ref2)
        #     resize_gt = transforms.Resize([100, 100])(resize_gt)
        # if self.img_transform is not None:
        #     exemplar = self.img_transform(exemplar)
        #     ref1 = self.img_transform(ref1)
        #     ref2 = self.img_transform(ref2)
        #     # exemplar_resize = self.img_transform(exemplar_resize)
        #     # ref1_resize = self.img_transform(ref1_resize)
        #     # ref2_resize = self.img_transform(ref2_resize)
        #     exemplar_resize = transforms.ToTensor()(exemplar_resize)
        #     ref1_resize = transforms.ToTensor()(ref1_resize)
        #     ref2_resize = transforms.ToTensor()(ref2_resize)
        # if self.target_transform is not None:
        #     exemplar_gt = self.target_transform(exemplar_gt)
        #     ref1_gt = self.target_transform(ref1_gt)
        #     ref2_gt = self.target_transform(ref2_gt)
        #     resize_gt = self.target_transform(resize_gt)
        # print(resize_gt.shape)
        # sample = {'exemplar': exemplar, 'exemplar_gt': exemplar_gt, 'ref1': ref1, 'ref1_gt': ref1_gt,
        #           'ref2': ref2, 'ref2_gt': ref2_gt, 'height': h, 'width': w}
        sample = {'exemplar': input_clip, 'exemplar_gt': gt_clip, 'uncertainty': uncertainty_clip, 'exemplar_resize': resize_clip,
                  'height': h_clip, 'width': w_clip, 'exemplar_path': path_clip}
        # , 'resize_gt': resize_gt
        return sample

    def generateImgFromVideo(self, root, is_train):
        imgs = []
        root = root[0]  # assume that only one video dataset
        if is_train:
            video_list = os.listdir(os.path.join(root[0], self.input_folder))
        else:
            video_list = validation_list
        for video in video_list:
            img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root[0], self.input_folder, video)) if f.endswith(self.img_ext)] # no ext
            img_list = self.sortImg(img_list)
            for img in img_list:
                # videoImgGt: (img, gt, video start index, video length)
                videoImgGt = (os.path.join(root[0], self.input_folder, video, img + self.img_ext),
                        os.path.join(root[0], self.label_folder, video, img + self.label_ext),
                        os.path.join(root[0], self.label_folder, video, img + '_uncertain' + self.label_ext),
                              self.num_video_frame, len(img_list))
                imgs.append(videoImgGt)
            self.num_video_frame += len(img_list)
        return imgs

    def generateImgFromSingle(self, root):
        imgs = []
        for sub_root in root:
            tmp = self.generateImagePair(sub_root[0])
            imgs.extend(tmp)  # deal with image case
            print('Image number of ImageSet {} is {}.'.format(sub_root[2], len(tmp)))

        return imgs

    def generateImagePair(self, root):
        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, self.input_folder)) if f.endswith(self.img_ext)]
        if len(img_list) == 0:
            raise IOError('make sure the dataset path is correct')
        return [(os.path.join(root, self.input_folder, img_name + self.img_ext), os.path.join(root, self.label_folder, img_name + self.label_ext))
            for img_name in img_list]

    def sortImg(self, img_list):
        img_int_list = [int(f) for f in img_list]
        sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
        return [img_list[i] for i in sort_index]

    def split_root(self, root):
        if not isinstance(root, list):
            raise TypeError('root should be a list')
        img_root_list = []
        video_root_list = []
        for tmp in root:
            if tmp[1] == 'image':
                img_root_list.append(tmp)
            elif tmp[1] == 'video':
                video_root_list.append(tmp)
            else:
                raise TypeError('you should input video or image')
        return img_root_list, video_root_list

    def __len__(self):
        return len(self.videoImg_list)//2*2


class VSD_Dataset(data.Dataset):
    def __init__(self, root, joint_transform=None, img_transform=None, target_transform=None, is_train = True):
        self.img_root, self.video_root = self.split_root(root)
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.input_folder = 'images'
        self.label_folder = 'labels'
        self.img_ext = '.jpg'
        self.label_ext = '.png'
        self.num_video_frame = 0
        # get all frames from video datasets
        self.videoImg_list = self.generateImgFromVideo(self.video_root, is_train)
        print('Total video frames is {}.'.format(self.num_video_frame))
        # get all frames from image datasets
        if len(self.img_root) > 0:
            self.singleImg_list = self.generateImgFromSingle(self.img_root)
            print('Total single image frames is {}.'.format(len(self.singleImg_list)))

    def __getitem__(self, index):
        manual_random = random.random()  # random for transformation
        # pair in video
        # pdb.set_trace()
        exemplar_path, exemplar_gt_path, videoStartIndex, videoLength = self.videoImg_list[index]  # exemplar
        clip_length = 6
        clip_index = [index + i for i in range(clip_length)]
        # ref_index1 = index-1
        # ref_index2 = index+1
        # if ref_index1 < videoStartIndex:
        # if index == videoStartIndex:
        #     ref_index1 = videoStartIndex+1
        #     ref_index2 = videoStartIndex+2
        # if ref_index2 > videoStartIndex+videoLength-1:
        #     ref_index1 = videoStartIndex + videoLength - 2
        #     ref_index2 = videoStartIndex + videoLength - 1
        if index + clip_length > videoStartIndex + videoLength:
            clip_index = [videoStartIndex + videoLength - 6 + i for i in range(clip_length)]
            # ref_index1 = videoStartIndex+videoLength-2
            # ref_index2 = videoStartIndex+videoLength-1

        # ref_index1 = index - 5
        # ref_index2 = index + 5
        # if ref_index1 < videoStartIndex:
        # # if index == videoStartIndex:
        #     ref_index1 = videoStartIndex + 5
        #     ref_index2 = videoStartIndex + 10
        # if ref_index2 > videoStartIndex+videoLength-1:
        #     ref_index1 = videoStartIndex + videoLength - 5
        #     ref_index2 = videoStartIndex + videoLength - 10
        # # elif index == videoStartIndex + videoLength - 1:
        # #     ref_index1 = videoStartIndex + videoLength - 2
        # #     ref_index2 = videoStartIndex + videoLength - 1

        # sample from same video
        clip_info = []
        for i in range(clip_length):
            path, gt_path, start_index, length = self.videoImg_list[clip_index[i]]
            clip_info.append([path, gt_path, start_index, length])
        # ref1_path, ref1_gt_path, videoStartIndex2, videoLength2 = self.videoImg_list[ref_index1]
        # ref2_path, ref2_gt_path, videoStartIndex3, videoLength3 = self.videoImg_list[ref_index2]

        # print(exemplar_path + ' ' + ref1_path + ' ' + ref2_path)
        for i in range(clip_length):
            if videoStartIndex != clip_info[i][3]:
                raise TypeError('wrong')
        # single image in image dataset

        # read image and gt and uncertainty
        input_clip = []
        gt_clip = []
        resize_clip = []
        path_clip = []
        h_clip = []
        w_clip = []
        for i in range(clip_length):
            path_clip.append(clip_info[i][0])
            exemplar = Image.open(clip_info[i][0]).convert('RGB')
            h_clip.append(exemplar.size[1])
            w_clip.append(exemplar.size[0])
            exemplar_gt = Image.open(clip_info[i][1]).convert('L')
            if self.joint_transform is not None:
                exemplar, exemplar_gt = self.joint_transform(exemplar, exemplar_gt, manual_random)
                exemplar_resize = transforms.Resize([384, 384])(exemplar)
                exemplar = self.img_transform(exemplar)
                exemplar_resize = transforms.ToTensor()(exemplar_resize)
                exemplar_gt = self.target_transform(exemplar_gt)
            input_clip.append(exemplar)
            gt_clip.append(exemplar_gt)
            resize_clip.append(exemplar_resize)

        # exemplar = Image.open(clip_info[0][0]).convert('RGB')
        # ref1 = Image.open(ref1_path).convert('RGB')
        # ref2 = Image.open(ref2_path).convert('RGB')
        # exemplar_gt = Image.open(exemplar_gt_path).convert('L')
        # ref1_gt = Image.open(ref1_gt_path).convert('L')
        # ref2_gt = Image.open(ref2_gt_path).convert('L')
        #
        # # print(exemplar.size)
        #
        #
        #
        # # transformation
        # if self.joint_transform is not None:
        #     # resize_gt = transforms.Resize([50, 50])(exemplar_gt)
        #     resize_gt = exemplar_gt
        #     exemplar, exemplar_gt = self.joint_transform(exemplar, exemplar_gt, manual_random)
        #     ref1, ref1_gt = self.joint_transform(ref1, ref1_gt, manual_random)
        #     ref2, ref2_gt = self.joint_transform(ref2, ref2_gt, manual_random)
        #     exemplar_resize = transforms.Resize([384, 384])(exemplar)
        #     ref1_resize = transforms.Resize([384, 384])(ref1)
        #     ref2_resize = transforms.Resize([384, 384])(ref2)
        #     resize_gt = transforms.Resize([100, 100])(resize_gt)
        # if self.img_transform is not None:
        #     exemplar = self.img_transform(exemplar)
        #     ref1 = self.img_transform(ref1)
        #     ref2 = self.img_transform(ref2)
        #     # exemplar_resize = self.img_transform(exemplar_resize)
        #     # ref1_resize = self.img_transform(ref1_resize)
        #     # ref2_resize = self.img_transform(ref2_resize)
        #     exemplar_resize = transforms.ToTensor()(exemplar_resize)
        #     ref1_resize = transforms.ToTensor()(ref1_resize)
        #     ref2_resize = transforms.ToTensor()(ref2_resize)
        # if self.target_transform is not None:
        #     exemplar_gt = self.target_transform(exemplar_gt)
        #     ref1_gt = self.target_transform(ref1_gt)
        #     ref2_gt = self.target_transform(ref2_gt)
        #     resize_gt = self.target_transform(resize_gt)
        # print(resize_gt.shape)
        # sample = {'exemplar': exemplar, 'exemplar_gt': exemplar_gt, 'ref1': ref1, 'ref1_gt': ref1_gt,
        #           'ref2': ref2, 'ref2_gt': ref2_gt, 'height': h, 'width': w}
        sample = {'exemplar': input_clip, 'exemplar_gt': gt_clip, 'exemplar_resize': resize_clip,
                  'height': h_clip, 'width': w_clip, 'exemplar_path': path_clip}
        # , 'resize_gt': resize_gt
        return sample

    def generateImgFromVideo(self, root, is_train):
        imgs = []
        root = root[0]  # assume that only one video dataset
        if is_train:
            video_list = os.listdir(os.path.join(root[0], self.input_folder))
        else:
            video_list = validation_list
        for video in video_list:
            img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root[0], self.input_folder, video)) if f.endswith(self.img_ext)] # no ext
            img_list = self.sortImg(img_list)
            for img in img_list:
                # videoImgGt: (img, gt, video start index, video length)
                videoImgGt = (os.path.join(root[0], self.input_folder, video, img + self.img_ext),
                        os.path.join(root[0], self.label_folder, video, img + self.label_ext),
                              self.num_video_frame, len(img_list))
                imgs.append(videoImgGt)
            self.num_video_frame += len(img_list)
        return imgs

    def generateImgFromSingle(self, root):
        imgs = []
        for sub_root in root:
            tmp = self.generateImagePair(sub_root[0])
            imgs.extend(tmp)  # deal with image case
            print('Image number of ImageSet {} is {}.'.format(sub_root[2], len(tmp)))

        return imgs

    def generateImagePair(self, root):
        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, self.input_folder)) if f.endswith(self.img_ext)]
        if len(img_list) == 0:
            raise IOError('make sure the dataset path is correct')
        return [(os.path.join(root, self.input_folder, img_name + self.img_ext), os.path.join(root, self.label_folder, img_name + self.label_ext))
            for img_name in img_list]

    def sortImg(self, img_list):
        img_int_list = [int(f) for f in img_list]
        sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
        return [img_list[i] for i in sort_index]

    def split_root(self, root):
        if not isinstance(root, list):
            raise TypeError('root should be a list')
        img_root_list = []
        video_root_list = []
        for tmp in root:
            if tmp[1] == 'image':
                img_root_list.append(tmp)
            elif tmp[1] == 'video':
                video_root_list.append(tmp)
            else:
                raise TypeError('you should input video or image')
        return img_root_list, video_root_list

    def __len__(self):
        return len(self.videoImg_list)//2*2
