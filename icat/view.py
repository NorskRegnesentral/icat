__author__      = "Anders U. Waldeland"
__copyright__   = "Copyright 2025, Norsk Regnesentral"
from collections.abc import Iterable

import numpy as np
from dash import html
import plotly.graph_objects as go


STATE_UNLABELLED = -1


STATIC_IMAGE_ROUTE = '/static/'


DEFAULT_COLORS = [
    "Red",
    "Lime",
    "Fuchsia",
    'chocolate',
    "Maroon",
    "Olive",
    "Green",
    "Navy",
    "Purple",
    'coral',
    'darkkhaki',
    'darkorange',
    'darkslateblue',
    'dodgerblue',
    'greenyellow',
    'hotpink',
    'indianred',
    'lightsalmon',
    'lightslategrey',
    'mediumturquoise',
    'orangered',
    'plum',
]
COLOR_UNLABELLED='Blue'

def css_for_image_border(image_class, is_selected, colors):
    if isinstance(image_class, Iterable):
        return [css_for_image_border(i, i_s, colors) for i, i_s in zip(image_class, is_selected)]

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
                "outline": "3px {} solid".format(colors[image_class%len(colors)]),
                "outlineOffset": "-5px",
            }


    elif image_class == STATE_UNLABELLED:
        return {
            "margin": "3px",
            }

    else:
        return {
            "margin": "3px",
            "outline": "3px {} solid".format(colors[image_class%len(colors)]),
            "outlineOffset": "-3px",
    }


def html_for_visible_images(index, data_object, zoom_value, hide_labelled, colors):

    data_object.n_times_img_clicked *= 0


    if isinstance(index, Iterable):
        return [html_for_visible_images(i, data_object, zoom_value, hide_labelled, colors) for i in index]

    image_path = data_object.path_to_images[index]
    image_class = data_object.get_class_label(index)
    is_selected = data_object.is_img_selected(index)

    # if hide_labelled and image_class != STATE_UNLABELLED:
    #     data_object.img_selected[index] = False
    #     return

    return html.Img(
        src=STATIC_IMAGE_ROUTE + image_path.split('/')[-1],
        width=zoom_value,
        id={'role': 'img', 'index': index},
        style=css_for_image_border(image_class, is_selected, colors),
        title=image_path
    )

def get_dropdown_options_for_labels(classes):
    return [ {'label': cls, 'value': i} for i,cls in enumerate(classes)] + [{'label':'Unlabelled', 'value':STATE_UNLABELLED}]



def get_scatter_plot_fig(data_object, category_to_show, size_labelled, size_unlabelled, colors, zoom):
    data_object.unselect_all()
    if category_to_show == -2:
        mask = np.ones_like(data_object.class_state, dtype='bool')
    else:
        mask = data_object.class_state == category_to_show

    marker_color = [
        COLOR_UNLABELLED.lower() if data_object.class_state[i] == STATE_UNLABELLED else colors[data_object.class_state[i]% len(colors)].lower()
        for i in np.where(mask)[0]
    ]

    marker_size = [
        size_unlabelled if data_object.class_state[i] == STATE_UNLABELLED else size_labelled
        for i in np.where(mask)[0]
    ]

    data_object.inds_of_imgs_in_scatter = np.where(mask)[0]
    scatter_plot = go.Scattergl(
        x=data_object.x[mask],
        y=data_object.y[mask],
        mode='markers',
        marker={'color': marker_color, 'size':marker_size ,'line': {'width':0}}
    )
    scatter_plot_figure = go.FigureWidget([scatter_plot])
    scatter_plot_figure.update_layout(clickmode='event+select') #Make scatterplot selectable

    return scatter_plot_figure
