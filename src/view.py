from collections.abc import Iterable

from dash import html

STATE_UNLABELLED = -1


STATIC_IMAGE_ROUTE = '/static/'


COLORS = [
    "Red",
    "Lime",
    "Blue",
    "Fuchsia",
    "Maroon",
    "Olive",
    "Green",
    "Teal",
    "Navy",
    "Purple"]


def css_for_image_border(image_class, is_selected):
    if isinstance(image_class, Iterable):
        return [css_for_image_border(i,i_s) for i, i_s in zip(image_class, is_selected)]

    if is_selected:
        if image_class == STATE_UNLABELLED:
            return {
                "border": "2px blue dashed",
                "margin": "1px",
            }
        else:
            return {
                "border": "2px blue dashed",
                "margin": "1px",
                "outline": "3px {} solid".format(COLORS[image_class]),
                "outlineOffset": "-5px",
            }


    elif image_class == STATE_UNLABELLED:
        return {
            "margin": "3px",
            }

    else:
        return {
            "margin": "3px",
            "outline": "3px {} solid".format(COLORS[image_class]),
            "outlineOffset": "-3px",
    }


def html_for_visible_images(index, data_object, zoom_value, hide_labelled):

    data_object.n_times_img_clicked *= 0


    if isinstance(index, Iterable):
        return [html_for_visible_images(i, data_object, zoom_value, hide_labelled) for i in index]

    image_path = data_object.path_to_images[index]
    image_class = data_object.get_class_label(index)
    is_selected = data_object.is_img_selected(index)

    if hide_labelled and image_class != STATE_UNLABELLED:
        data_object.img_selected[index] = False
        return

    return html.Img(
        src=STATIC_IMAGE_ROUTE + image_path.split('/')[-1],
        width=zoom_value,
        id={'role': 'img', 'index': index},
        style=css_for_image_border(image_class, is_selected),
    )

def get_dropdown_options_for_labels(classes):
    return [ {'label': cls, 'value': i} for i,cls in enumerate(classes)]

##

