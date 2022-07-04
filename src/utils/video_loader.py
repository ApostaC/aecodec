import os, sys
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, root_dir, frame_size=None):
        self._dataset_dir = os.path.join(root_dir)
        self._frame_size = frame_size
        self._total_frames = 0 # Storing file names in object

        self.get_file_names()
        self._num_files = len(self.__file_names)

        self.reset()

    def reset(self):
        self._curr_counter = 0
        self._frame_counter = -1 # Count the number of frames used per file
        self._file_counter = -1 # Count the number of files used
        self._dataset_nums = [] # Number of frames to be considered from each file (records+files)
        self._clip = [] # hold video frames
        self._cur_file_names = list(self.__file_names)

    @property
    def data(self):
        self._curr_counter+=1
        return self.__getitem__(self._curr_counter)

    def __getitem__(self, idx):
        # Get the next dataset if frame number is more than table count
        if not len(self._dataset_nums) or self._frame_counter >= self._dataset_nums[self._file_counter]-1:
            self.current_file = self._cur_file_names.pop() # get one filename
            cap = cv2.VideoCapture(self.current_file)
            # Check if camera opened successfully
            if (cap.isOpened()== False):
                print("Error opening video stream or file")
            # Read until video is completed
            self._clip = []
            while(cap.isOpened()):
                # Capture frame-by-frame
                ret, img = cap.read()
                if ret != True:break
                # skip black frames
                if np.sum(img) == 0:continue
                img = Image.fromarray(img)
                if self._frame_size is not None:
                    img = img.resize(self._frame_size)
                self._clip.append(img)
                if len(self._clip) % 200 == 0:
                    print("Got {} frames...".format(len(self._clip)))
                if len(self._clip) > 1000:
                    break
            self._file_counter +=1
            self._dataset_nums.append(len(self._clip))
            self._frame_counter = 0
        else:
            self._frame_counter+=1
        return self._clip[self._frame_counter],self._frame_counter==self._dataset_nums[self._file_counter]-1

    def get_file_names(self):
        print("[log] Looking for files in", self._dataset_dir)
        self.__file_names = []
        for fn in os.listdir(self._dataset_dir):
            fn = fn.strip("'")
            if fn.split('.')[-1] == 'mp4':
                self.__file_names.append(self._dataset_dir + '/' + fn)
                break # only test one video
        print("[log] Number of files found {}".format(len(self.__file_names)))

    def __len__(self):
        if not self._total_frames:
            self.count_frames()
        return self._total_frames


    def count_frames(self):
        # Count total frames
        self._total_frames = 0
        for file_name in self.__file_names:
            cap = cv2.VideoCapture(file_name)
            # Check if camera opened successfully
            if (cap.isOpened()== False):
                print("Error opening video stream or file")
            # Read until video is completed
            while(cap.isOpened()):
                # Capture frame-by-frame
                ret, img = cap.read()
                if ret != True:break
                if np.sum(img) == 0:continue
                self._total_frames+=1
            # When everything done, release the video capture object
            cap.release()
        print("[log] Total frames: ", self._total_frames)

if __name__ == "__main__":
    test_dataset = VideoDataset("../../data/videos/", frame_size=(256, 256))
    test_dataset.count_frames()
    for idx in range(700):
        frame, eof = test_dataset[idx]
        print(idx, frame, eof)
