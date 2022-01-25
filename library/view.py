from collections.abc import Iterable

from dash import html

STATE_UNLABELLED = -1


STATIC_IMAGE_ROUTE = '/static/'


COLORS = [
    "Red",
    "Yellow",
    "Lime",
    "Aqua",
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
        return {"outline": "3px blue dashed", "outline-offset": "-3px", "margin": "2px"}

    elif image_class == STATE_UNLABELLED:
        return {"outline": "0px blue dashed", "margin": "2px"}

    else:
        return {"outline": "3px {} solid".format(COLORS[image_class]), "outline-offset": "-3px", "margin": "2px"}


def html_for_visible_images(index, image_path, zoom_value):
    if isinstance(index, Iterable):
        return [html_for_visible_images(i, ip, zoom_value) for i, ip in zip(index, image_path)]

    return html.Img(src=STATIC_IMAGE_ROUTE + image_path, width=zoom_value, id={'role': 'img', 'index': index}, style={"margin": "2px"})

def get_dropdown_options_for_labels(classes):
    return [ {'label': cls, 'value': i} for i,cls in enumerate(classes)]
