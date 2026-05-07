import glob
import re
import os.path as osp

from .bases import BaseImageDataset

class CustomMSMT17(BaseImageDataset):
    dataset_dir = "MSMT17_V1"

    def __init__(self, root="", verbose=True, pid_begin=0, **kwargs):
        super(CustomMSMT17, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")

        self.list_train_path = osp.join(self.dataset_dir, "list_train.txt")
        self.list_val_path = osp.join(self.dataset_dir, "list_val.txt")
        self.list_query_path = osp.join(self.dataset_dir, "list_query.txt")
        self.list_gallery_path = osp.join(self.dataset_dir, "list_gallery.txt")

        self._check_before_run()
        train = self._process_dir(self.train_dir, self.list_train_path)
        val = self._process_dir(self.train_dir, self.list_val_path)
        train += val
        query = self._process_dir(self.query_dir, self.list_query_path)
        gallery = self._process_dir(self.gallery_dir, self.list_gallery_path)

        if verbose: 
            print("=> Custom MSMT17 is loaded")

            self.print_dataset_statistics(train, query, gallery)

        
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
    
    def _process_dir(self, dir_path, list_path):
        if not osp.exists(list_path):
            print(f"[ERROR] {list_path} not found!")
            return []
        
        with open(list_path, "r") as txt:
            lines = txt.readlines()

            dataset = []
            pid_container = set()
            cam_container = set()
            
            for img_info in lines:
                img_path = img_info.strip()
                if not img_path:
                    print(f"[WARNING] {img_path}")
                    continue
                
                parts = img_path.split("_")
                pid = int(parts[0])
                camid = int(parts[1].replace("c", ""))
                
                img_path = osp.join(dir_path, img_path)

                dataset.append((img_path, self.pid_begin+pid, camid-1, 1))

                pid_container.add(pid)
                cam_container.add(camid)
            
            for idx, pid in enumerate(pid_container):
                assert idx == pid, "[ERROR] PID not start from 0 or not increments with 1"
            
            return dataset
    
    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"[ERROR] {self.dataset_dir} is not available")
        if not osp.exists(self.train_dir):
            raise RuntimeError(f"[ERROR] {self.train_dir} is not available")
        if not osp.exists(self.query_dir):
            raise RuntimeError(f"[ERROR] {self.query_dir} is not available")
        if not osp.exists(self.gallery_dir):
            raise RuntimeError(f"[ERROR] {self.gallery_dir} is not available")

class CustomMarket1501(BaseImageDataset):
    dataset_dir = "Market1501"

    def __init__(self, root="", verbose=True, pid_begin=0, **kwargs):
        super(CustomMarket1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Custom Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)
        
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
    
    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"[ERROR] {self.dataset_dir} is not available")
        if not osp.exists(self.query_dir):
            raise RuntimeError(f"[ERROR] {self.query_dir} is not available")
        if not osp.exists(self.train_dir):
            raise RuntimeError(f"[ERROR] {self.train_dir} is not available")
        if not osp.exists(self.gallery_dir):
            raise RuntimeError(f"[ERROR] {self.gallery_dir} is not found")
    
    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))

        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()

        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())

            if pid == -1: continue

            pid_container.add(pid)
        
        pid2label = {
            pid: label for label, pid in enumerate(pid_container)
        }

        dataset = []

        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())

            if pid == -1: continue

            assert 0 <= pid <= 1501
            assert 1 <= camid <= 6

            camid -= 1
            if relabel:
                pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin+pid, camid, 1))

        return dataset