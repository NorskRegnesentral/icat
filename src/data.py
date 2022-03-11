from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime

import numpy as np

STATE_UNLABELLED = -1

class ImageClusterData():

    def __init__(self, path_to_images, x, y, classes, labels):
        if classes is None:
            classes = []
        self.classes = classes
        self.path_to_images = path_to_images
        self.x = x
        self.y = y
        self.class_state = np.zeros(len(x), dtype='int16') + STATE_UNLABELLED
        self.n_times_img_clicked = np.zeros(len(x), dtype='uint64')
        self.img_selected  =np.zeros(len(x), dtype='bool')
        self.inds_of_imgs_in_scatter = None
        now = datetime.now()

        if labels is not None:
            try:
                for line in labels:
                    if line.startswith('#'):
                        continue
                    file,cls = line.split(';')
                    cls = int(cls.strip(' '))
                    try:
                        ind = self.path_to_images.index(file)
                    except ValueError: #File is not in list
                        continue
                    self.class_state[ind] = cls
            except Exception as e:
                print('Failed to parse label-file')
                raise e
        self.labels = labels



    def get_mscoco(self):

        now = datetime.now()
        info =  '# Exported from iCat on {} with {} annotations'.format(
                now.strftime("%m/%d/%Y, %H:%M:%S"),
                int(np.sum(self.class_state!=STATE_UNLABELLED))
        )
        class_str = '# classes = ["' + '","'.join(self.classes) + '"]'
        return [info] + [class_str] + ["{}; {}".format(self.path_to_images[i], int(self.class_state[i])) for i in range(len(self)) if self.class_state[i]!=STATE_UNLABELLED]

    def set_class_of_selected(self, new_class_label):
        self.class_state[self.img_selected] = new_class_label

    def get_class_label(self, indexes):
        return self.class_state[np.array(indexes)]

    def export_mscoco(self):
        raise NotImplementedError()

    def update_if_img_was_clicked(self, index, n_clicks):
        if isinstance(index, Iterable):
            return [self.update_if_img_was_clicked(i, nc) for i, nc in zip(index, n_clicks)]

        if n_clicks is None:
            return False

        elif n_clicks > self.n_times_img_clicked[index]:
            self.n_times_img_clicked[index] = n_clicks
            self.toggle_selected_state(index)
            return True

        else:
            return False

    def is_img_selected(self, index):
        if isinstance(index, Iterable):
            return [self.is_img_selected(i) for i in index]
        if not len(self.classes):
            return False
        return self.img_selected[index]

    def toggle_selected_state(self, index):
        if isinstance(index, Iterable):
            return [self.is_img_selected(i) for i in index]
        self.img_selected[index] = not self.img_selected[index]

    def select_img(self, index):
        if isinstance(index, Iterable):
            return [self.select_img(i) for i in index]
        if not len(self.classes):
            return
        self.img_selected[index] = True

    def unselect_all(self):
        self.img_selected[:] = False

    def __len__(self):
        return len(self.x)