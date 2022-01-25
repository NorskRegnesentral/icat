from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime

import numpy as np

STATE_UNLABELLED = -1

class ImageClusterData():

    def __init__(self, path_to_images, x, y, classes, mscoco_dict):
        if classes is None:
            classes = []
        self.classes = classes
        self.path_to_images = path_to_images
        self.x = x
        self.y = y
        self.class_state = np.zeros(len(x), dtype='int16') + STATE_UNLABELLED
        self.n_times_img_clicked = np.zeros(len(x), dtype='uint64')
        self.img_selected  =np.zeros(len(x), dtype='bool')

        now = datetime.now()

        if mscoco_dict is None:
            mc = {}
        else:
            mc = mscoco_dict

        #Get classes from mscoco dict
        if not len(self.classes) and len(mscoco_dict.get('categories',{})):
            class_numbers = [m['id'] for m in mscoco_dict['categories']]
            class_names = [m['name'] for m in mscoco_dict['categories']]
            self.classes = [class_names[i] for i in sorted(class_numbers)]

        #Make categories list
        if not 'categories' in mc:
            mc['categories'] = [
                {'id': i, 'name':c}
                for i,c in enumerate(self.classes)
            ]

        #Make image-list
        if not 'images' in mc:
            mc['images'] = []

            for f in self.path_to_images:
                mc['images'].append(
                    {
                        "file_name": f,
                        "id": f
                    }
                )
        if not 'annotations' in mc:
            mc['annotations'] = []

        if not 'info' in mc:
            mc['info'] = {}

        if "description" not in mc['info']:
            mc['info']["description"] = "Labels generated with iCat"
        if "date_created" not in mc['info']:
            mc['info']["date_created"] = now.strftime("%m/%d/%Y, %H:%M:%S")

        if "change_log" not in mc['info']:
            mc['info']["change_log"] = []

        if mscoco_dict is not None:
            mc['info']["change_log"].append(
                'Imported into iCat on {} with {} existing annotations'.format(
                    now.strftime("%m/%d/%Y, %H:%M:%S"),
                    len(mc['annotations'])
                )
            )
        self.mc = mc

    def get_mscoco(self):
        mc = deepcopy(self.mc)
        for i, ind in enumerate(np.where(self.class_state>-1)[0]):
            mc['annotations'].append(
                {
                    'id': i,
                    'image_id': str(self.path_to_images[ind]),
                    'category_id': int(self.class_state[ind])

                }
            )
        now = datetime.now()
        mc['info']["change_log"].append(
            'Exported from iCat on {} with {} annotations'.format(
                now.strftime("%m/%d/%Y, %H:%M:%S"),
                i)
        )
        return mc

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
        return self.img_selected[index]

    def toggle_selected_state(self, index):
        if isinstance(index, Iterable):
            return [self.is_img_selected(i) for i in index]
        self.img_selected[index] = not self.img_selected[index]

    def select_img(self, index):
        if isinstance(index, Iterable):
            return [self.select_img(i) for i in index]
        self.img_selected[index] = True

    def unselect_all(self):
        self.img_selected[:] = False

