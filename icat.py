import json

import flask
import numpy as np

from dash import dcc
from dash import html
from dash_extensions.enrich import Output, DashProxy, Input, State, ALL, MultiplexerTransform
import plotly.graph_objects as go

from library.data import ImageClusterData
from library.utils import downsample_to_N
from library.view import get_dropdown_options_for_labels, html_for_visible_images, css_for_image_border, STATIC_IMAGE_ROUTE


################################################################################
# Initialize data-object

cluster_file = np.load('/Users/andersuw/Desktop/ballast_dataset/tsne_BYOL_21.jan.npz')
img_folder = '/Users/andersuw/Desktop/ballast_dataset/v01/'
image_paths = cluster_file['files']
image_paths = [f.split('/')[-1] for f in image_paths]
xy = cluster_file['xy']
x = xy[:, 0]
y = xy[:, 1]

# inds = np.where(np.bitwise_and(x>20,y>20))[0]
# x = x[inds]
# y = y[inds]
# image_paths = [image_paths[i] for i in inds]
#Downsample number of points
# max_N = 10000
# x = downsample_to_N(x, max_N)
# y = downsample_to_N(y, max_N)
# image_paths = downsample_to_N(image_paths, max_N)

max_selected=100

data = ImageClusterData(image_paths, x, y, ['ok', 'for små korn', 'grus/sand'], None)
image_zoom_value = 100
selected_class = None
all_is_selected = False
app = DashProxy(__name__, prevent_initial_callbacks=True, transforms=[MultiplexerTransform()])

##############################################################################
# LAYOUT OF WEBPAGE
scatter_plot_module = go.FigureWidget([go.Scattergl(x=x, y=y, mode='markers')])

app.layout = html.Div(children=[
    html.H1(children='iCAT - image Cluster Analysis Tool'),

    html.Div(children='''

    '''),

    html.Div([
        html.Div(
            [
                html.Div(id='slider-text', style={'text-align': 'center'}),
                dcc.Slider(
                    id='im-size-slider',
                    min=0,
                    max=400,
                    step=0.5,
                    value=image_zoom_value,
                ),
            ]
        ),
        html.Div(
            [
                html.Button('Download mscoco', id='download_button'),
                dcc.Download(id="download-text"),
                html.Button('Hide/un-hide labelled ', id='hide_labelled_button'),
                html.Button('Select/Un-select all', id='select_all_button'),
                html.Button('Label selected as:', id='label_button'),
                dcc.Dropdown(
                    id='label_list',
                    options=get_dropdown_options_for_labels(data.classes),
                    clearable=False,
                    value=data.classes[0]
                )
            ],
        ),
    ],
        style={'width': "49%", 'float': 'right'}
    ),

    html.Div([

        html.Div(
            [
                dcc.Graph(
                    id='scatter_plot',
                    figure=scatter_plot_module)
            ],
            style={'width': '49%', 'float': 'left'}
        ),
        html.Div(
            [

                html.Div(id='image_field')
            ],
            style={'width': '49%', 'float': 'right'}
        ),
    ]),
])

scatter_plot_module.update_layout(clickmode='event+select', height=1000) #Make scatterplot selectable

##############################################################################
# CALLBACKS ON GUI INTERACTION
# STATE variables for GUI


# Select points in scatter
@app.callback(
    Output('image_field', 'children'),
    Output({'role': 'img', 'index': ALL}, 'style'),
    Input('scatter_plot', 'selectedData'),
    State({'role': 'img', 'index': ALL}, 'id')
)
def points_marked_in_cluster(marked_scatter_points, ids):
    # Returns
    # - html for images that should be displayed
    # - css for border around images
    global data, all_is_selected
    data.unselect_all()
    all_is_selected = False

    if marked_scatter_points is None:
        return '', [None] * len(ids)

    indexes_of_marked = [p['pointNumber'] for p in marked_scatter_points['points']]
    indexes_of_marked = downsample_to_N(indexes_of_marked, max_selected, random=True)
    print('Selected {} points in scatter.'.format(len(indexes_of_marked)))
    path_to_images = [data.path_to_images[i] for i in indexes_of_marked]

    all_is_selected = True
    data.select_img(indexes_of_marked)

    image_htmls = html_for_visible_images(indexes_of_marked, path_to_images, image_zoom_value )
    image_css = css_for_image_border(
        data.get_class_label(indexes_of_marked),
        data.is_img_selected(indexes_of_marked)
    )
    print('images + css sendt')
    return image_htmls, image_css


# On image click
@app.callback(
    Output({'role': 'img', 'index': ALL}, 'style'),
    [Input({'role': 'img', 'index': ALL}, 'n_clicks')],
    State({'role': 'img', 'index': ALL}, 'id'),
    prevent_initial_call=True,
)
def image_onclick(n_clicks, image_elements):
    # On click on image return
    # - CSS for image
    print('Clicked on an image')
    indexes = [ie['index'] for ie in image_elements]

    data.update_if_img_was_clicked(indexes, n_clicks)

    image_css = css_for_image_border(
        data.get_class_label(indexes),
        data.is_img_selected(indexes)
    )
    return image_css


# On image-slider value change
@app.callback(Output('slider-text', 'children'),
              Input('im-size-slider', 'value'))
def display_value(value):
    global image_zoom_value, visible_imgs
    print('Changing zoom value to {}'.format(value))
    image_zoom_value = value

    # for img in visible_imgs:
    #     img.height=value

    return 'Zoom: ' + str(value)


# On label_list dropdown-menu change
@app.callback(Output('label_list', 'value'),
              Input('label_list', 'value'))
def set_selected_class(value):
    print('Setting selected class to {}'.format(value))
    global selected_class
    selected_class = value
    return value


# On label_selected button click
@app.callback(Output({'role': 'img', 'index': ALL}, 'style'),
              Input('label_button', 'n_clicks'),
              State({'role': 'img', 'index': ALL}, 'id'),
              )
def set_selected_class(n_clicks, image_elements):
    global selected_class
    indexes = [ie['index'] for ie in image_elements]
    print('Setting label', selected_class, 'for', indexes)
    if not len(indexes) or selected_class is None:
        return []

    data.set_class_of_selected(selected_class)
    data.unselect_all()
    all_is_selected = False
    image_css = css_for_image_border(
        data.get_class_label(indexes),
        data.is_img_selected(indexes)
    )

    return  image_css

    return n_clicks, border_output

# On select/unselect
@app.callback(Output({'role': 'img', 'index': ALL}, 'style'),
              Input('select_all_button', 'n_clicks'),
              State({'role': 'img', 'index': ALL}, 'id'),
              )
def toggle_all_selected(n_clicks, image_elements):
    print('toggle_all_selected')
    indexes = [ie['index'] for ie in image_elements]

    global all_is_selected
    if all_is_selected:
        all_is_selected = False
        data.unselect_all()
    else:
        all_is_selected = False
        indexes = [ie['index'] for ie in image_elements]
        data.select_img(indexes)

    image_css = css_for_image_border(
        data.get_class_label(indexes),
        data.is_img_selected(indexes)
    )
    print('css sendt')
    return  image_css

# Hide labelled images
@app.callback(Output({'role': 'img', 'index': ALL}, 'style'),
              Input('hide_labelled_button', 'n_clicks'),
              State({'role': 'img', 'index': ALL}, 'id'),
              )
def toggle_hide_labelled(n_clicks, image_elements):
    print('toggle_hide_labelled')
    indexes = [ie['index'] for ie in image_elements]

    data.unselect_all()
    all_is_selected = False

    if all_is_selected:
        all_is_selected = False
        data.unselect_all()
    else:
        all_is_selected = False
        indexes = [ie['index'] for ie in image_elements]
        data.select_img(indexes)

    image_css = css_for_image_border(
        data.get_class_label(indexes),
        data.is_img_selected(indexes)
    )

    return  image_css




@app.callback(
    Output("download-text", "data"),
    Input("download_button", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dict(content=json.dumps(data.get_mscoco(), indent=2), filename="labels.json")

# Function that serves images to webpage
# Add a static image route that serves images from desktop
# Be *very* careful here - you don't want to serve arbitrary files
# from your computer or server
@app.server.route('{}<image_path>'.format(STATIC_IMAGE_ROUTE))
def serve_image(image_path):
    # image_name = '{}.png'.format(image_path)
    if image_path not in image_paths:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(img_folder, image_path)


if __name__ == '__main__':
    app.run_server(debug=True, port=8030)
##

