import argparse
import json
import os
import time

import flask
import numpy as np

from dash import dcc
from dash import html
from dash.development.base_component import Component
from dash_extensions import DeferScript
from dash_extensions.enrich import Output, DashProxy, Input, State, ALL, MultiplexerTransform, MATCH
import plotly.graph_objects as go
from flask_caching import Cache

from src.data import ImageClusterData, STATE_UNLABELLED
from src.utils import downsample_to_N
from src.view import get_dropdown_options_for_labels, html_for_visible_images, css_for_image_border, \
    STATIC_IMAGE_ROUTE, COLORS




def run_icat(file, classes = [], replace_path=None, replace_part=None,  max_selected = 200, port=8030, mscoco=None):
    ################################################################################
    # Initialize data-object

    #Open data
    cluster_file = np.load(file)

    if type(mscoco) == str:
        with open(mscoco) as f:
            mscoco = json.load(f)

    #Check input
    assert 'files' in cluster_file, 'Could not find "files" in {}'.format(file)
    assert 'xy' in cluster_file, 'Could not find "xy" in {}'.format(file)
    # assert len(cluster_file['files']) == cluster_file['xy'].shape[0], 'there was a miss match between the length of "files" ({}) and the first shape component of "xy" {}'.format(len(cluster_file['files']), cluster_file['xy'].shape)


    image_paths = cluster_file['files']
    image_paths = [f.split('/')[-1] for f in image_paths]

    if replace_path is not None and not os.path.isdir(replace_path):
        raise Exception('Could not find folder {}'.format(replace_path))

    if replace_path is not None and replace_part is None:
        replace_path += '/'
        replace_path = replace_path.replace('//','/')
        image_paths = [f.split('/')[-1] for f in image_paths]
        image_paths = [os.path.join(replace_path, f) for f in image_paths]
    elif replace_path is not None and replace_part is not None:
        replace_path += '/'
        replace_path = replace_path.replace('//', '/')
        replace_part += '/'
        replace_part = replace_part.replace('//', '/')
        image_paths = [f.replace(replace_part, replace_path) for f in image_paths]

    files_that_can_be_found = [os.path.isfile(f) for f in image_paths]
    if np.mean(files_that_can_be_found) == 0:
        raise FileNotFoundError('Could not find files. Eg. {}'.format(image_paths[0]))
    assert np.mean(files_that_can_be_found)>.9, 'More than 10% of the files can not be found'

    xy = cluster_file['xy']

    if np.mean(files_that_can_be_found)!=1:
        print('WARNING: could not find all files - ICAT will remove ignore that are missing - {} of {} files found'.format(np.sum(files_that_can_be_found), len(files_that_can_be_found)))
        inds = np.where(files_that_can_be_found)[0]
        xy = xy[inds,:]
        image_paths = [image_paths[i] for i in inds]


    # State for this session
    global data, image_zoom_value,  selected_class, all_is_selected, hide_labelled_images
    data = ImageClusterData(image_paths, xy[:, 0],  xy[:, 1], classes, mscoco)
    image_zoom_value = 100
    selected_class = None
    all_is_selected = False
    hide_labelled_images = False
    app = DashProxy(__name__, prevent_initial_callbacks=True, transforms=[MultiplexerTransform()])

    app.scripts.config.serve_locally = True
    app.css.config.serve_locally = True

    ##############################################################################
    # LAYOUT OF WEBPAGE
    marker_color = 'blue' if mscoco is None else ['blue' if data.class_state[i] == STATE_UNLABELLED else COLORS[data.class_state[i]].lower() for i in range(len(data))]
    scatter_plot = go.Scattergl(x=xy[:, 0], y= xy[:, 1], mode='markers', marker={'color':marker_color})
    scatter_plot_figure = go.FigureWidget([scatter_plot])

    app.layout = html.Div(children=[



        html.Div([
            html.Div(
                [
                    html.Div(['Zoom: 100%'],id='slider-text', style={'textAlign': 'center'}),
                    dcc.Slider(
                        id='im-size-slider',
                        min=0,
                        max=400,
                        step=0.5,
                        value=image_zoom_value,
                    ),
                ], style={'width':'30%'}
            ),
            html.Div(
                [
                    html.Button('Download mscoco', id='download_button'),
                    dcc.Download(id="download-text"),
                    html.Button('Hide/un-hide labelled ', id='hide_labelled_button'),
                    html.Button('Select/Un-select all', id='select_all_button'),
                    html.Div(
                        [
                            html.Button('Label selected as:', id='label_button'),
                        dcc.Dropdown(
                            id='label_list',
                            options=get_dropdown_options_for_labels(data.classes),
                            clearable=False,
                            value=data.classes[0] if len(data.classes) else None,

                            )
                        ],
                        style={'float': 'right',  'width': '40%'}
                    ),
                ], hidden= len(data.classes)==0,#Hide this part if no labels are provided
            ),
            html.Div( [ html.Plaintext(c, style={'color':COLORS[i], 'width': str(int(50//len(data.classes)))+'%', 'margin':'2px'}) for i,c in enumerate(data.classes)])
        ],
            style={'width': "49%", 'float': 'right'}
        ),

        html.Div([
            html.H1(children='iCAT - image Cluster Analysis Tool'),

            html.Div(
                [
                    dcc.Graph(
                        id='scatter_plot',
                        figure=scatter_plot_figure)
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
        # DeferScript(src='/static/select_lasso.js') #This cause lasso.js being reloaded every 5 s - shoul only run once
    ])


    #app.scripts.append_script(lasso_select_script)
    scatter_plot_figure.update_layout(clickmode='event+select', height=1000) #Make scatterplot selectable
    scatter_plot
    ##############################################################################
    # CALLBACKS ON GUI INTERACTION
    # STATE variables for GUI


    # Select points in scatter
    @app.callback(
        Output('image_field', 'children'),
        Input('scatter_plot', 'selectedData'),
        State({'role': 'img', 'index': ALL}, 'id'),
        prevent_initial_call = True,

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

        all_is_selected = True
        data.select_img(indexes_of_marked)

        print('Selected {} points in scatter.'.format(len(indexes_of_marked)))

        return html_for_visible_images(indexes_of_marked, data, image_zoom_value, hide_labelled_images)


    # On image click
    @app.callback(
        Output({'role': 'img', 'index': MATCH}, 'style'),
        [Input({'role': 'img', 'index': MATCH}, 'n_clicks')],
        State({'role': 'img', 'index': MATCH}, 'id'),
        prevent_initial_call=True,
    )
    def image_onclick(n_clicks, image_element):
        # On click on image return
        # - CSS for image
        print('Clicked on an image')
        index = image_element['index']

        data.update_if_img_was_clicked(index, n_clicks)

        image_css = css_for_image_border(
            data.get_class_label(index),
            data.is_img_selected(index)
        )
        print('SEND')
        return image_css


    # On image-slider value change
    @app.callback(
        [Output('slider-text', 'children'),
        Output('image_field', 'children')],
        Input('im-size-slider', 'value'),
        State({'role': 'img', 'index': ALL}, 'id'),
    )
    def display_value(value,image_elements):
        global image_zoom_value, visible_imgs
        print('Changing zoom value to {}'.format(value))
        image_zoom_value = value

        indexes = [i['index'] for i in image_elements]
        return 'Zoom: ' + str(value) +'%',  html_for_visible_images(indexes, data, image_zoom_value, hide_labelled_images)



    # On label_list dropdown-menu change
    @app.callback(Output('label_list', 'value'),
                  Input('label_list', 'value'))
    def set_selected_class(value):
        print('Setting selected class to {}'.format(value))
        global selected_class
        selected_class = value
        return value


    # On label_selected button click
    @app.callback(Output('image_field', 'children'),
                  Input('label_button', 'n_clicks'),
                  State({'role': 'img', 'index': ALL}, 'id'),
                  prevent_initial_call=True,
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

        indexes = [p['index'] for p in image_elements]
        return html_for_visible_images(indexes, data, image_zoom_value, hide_labelled_images)


    # On select/unselect
    @app.callback(Output('image_field', 'children'),
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
            all_is_selected = True
            indexes = [ie['index'] for ie in image_elements]
            data.select_img(indexes)

        return html_for_visible_images(indexes, data, image_zoom_value, hide_labelled_images)


    # Hide labelled images
    @app.callback(Output('image_field', 'children'),
                  Input('hide_labelled_button', 'n_clicks'),
                  State({'role': 'img', 'index': ALL}, 'id'),
                  )
    def toggle_hide_labelled(n_clicks, image_elements):
        global hide_labelled_images

        print('toggle_all_selected')
        indexes = [ie['index'] for ie in image_elements]

        if hide_labelled_images:
            hide_labelled_images = False

        else:
            hide_labelled_images = True

        return html_for_visible_images(indexes, data, image_zoom_value, hide_labelled_images)



    @app.callback(
        Output("download-text", "data"),
        Input("download_button", "n_clicks"),
        prevent_initial_call=True,
    )
    def func(n_clicks):
        return dict(content=json.dumps(data.get_mscoco(), indent=2), filename="labels.json")

    # Function that serves files to webpage
    # Be *very* careful here - you don't want to serve arbitrary files
    # from your computer or server
    allowed_files_to_upload = [f.split('/')[-1] for f in data.path_to_images]
    @app.server.route('{}<file>'.format(STATIC_IMAGE_ROUTE))
    def serve_files(file):
        print('Client asking for:', file)
        if file == 'select_lasso.js':
            print('serving select_lasso.js')
            return flask.send_from_directory(os.path.join(os.path.dirname(__file__),'res'), 'select_lasso.js')
        elif file.split('/')[-1] in allowed_files_to_upload:
            ind = allowed_files_to_upload.index(file.split('/')[-1])
            path = data.path_to_images[ind]
            return flask.send_from_directory(os.path.dirname(path), file.split('/')[-1])
        else:
            raise Exception('"{}" is excluded from the allowed static files'.format(file))


    app.run_server(debug=True, port=port)

##


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ICAT - Image Cluster Annotation Tool')
    optionalNamed = parser.add_argument_group('Required arguments')
    optionalNamed.add_argument(
        '-f',
        '--file',
        help='A npz-file with two fields "xy", "files". "xy" should be a Nx2 array of floats with positions of the images in cluster space. The last should be a array of strings with path to the images.',
        required=True
    )
    optionalNamed = parser.add_argument_group('Optional arguments')
    optionalNamed.add_argument(
        '-c',
        '--classes',
        help='A comma separated list of classes (required for labelling)',
        default =[],
        required=False)

    optionalNamed.add_argument(
        '-r',
        '--replace_path',
        help='Replaces the path to the folder of the images.',
        default=None,
        required=False)

    optionalNamed.add_argument(
        '-rp',
        '--replace_part',
        help='If this is provided (and --replace_path is provided) only this part of the paths will be replaced with replace_path ',
        default=None,
        required=False)

    optionalNamed.add_argument(
        '-n' ,
        '--max_selected',
        help='How many images should be displayed at once (a large number here will make icat sluggish)',
        default=200,
        required=False)

    optionalNamed.add_argument(
        '-p' ,
        '--port',
        help='Port',
        default=8030,
        required=False)

    optionalNamed.add_argument(
        '-m' ,
        '--mscoco',
        help='mscoco-file for existing labels',
        default=None,
        required=False)

    args = parser.parse_args()


    run_icat(args.file,
             classes = args.classes,
             replace_path=args.replace_path,
             replace_part=args.replace_part,
             max_selected = args.max_selected,
             port=args.port,
             mscoco=args.mscoco)
