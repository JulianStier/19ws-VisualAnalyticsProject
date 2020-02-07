import os
import geopandas as gpd
import ipyleaflet
import ipywidgets as widgets
import matplotlib
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import shapely
from IPython.display import display
from ipywidgets import fixed, Layout, interactive_output
from matplotlib import pyplot as plt


def get_time_slider(df, timestamp_column='_timestamp'):
    def get_date_range():
        timestamps = pd.to_datetime(df[timestamp_column]).apply(pd.Timestamp.date)
        vmin, vmax = timestamps.min(), timestamps.max()
        return pd.Series(pd.date_range(vmin, vmax)).apply(pd.Timestamp.date)

    date_range = get_date_range()
    return widgets.SelectionRangeSlider(options=date_range.values, description='Date Range', continuous_update=False,
                                        index=(0, len(date_range) - 1), values=(0, len(date_range)),
                                        layout=Layout(width='500px'))


def get_nuts_shapes(shp_folder='nuts_data', simplify=False, tol=1e-3):
    def get_fns(directory, condition=lambda x: True):
        return list(filter(condition, [directory + '/' + fn for fn in os.listdir(directory)]))

    folders = get_fns(shp_folder, os.path.isdir)
    files = np.hstack([get_fns(folder, lambda f: f.endswith('.shp')) for folder in folders])
    geo_df = pd.concat(list(map(gpd.read_file, files)))
    if simplify:
        geo_df.geometry = geo_df.geometry.simplify(tol)
    return geo_df


def get_shapes_heatmap(data, nuts_ids_column, color_column, logarithmic: bool = False, cmap='viridis',
                       info_columns=('NUTS_ID', 'NUTS_NAME', 'num_persons'), info_widget_html=None,
                       vmin=0, vmax=1, time_hist=None, full_data=None, date_limits=None, tweets_table=None,
                       table_columns=('_timestamp', 'text_translated', 'num_persons', 'mode')):
    def get_layer(shapes: gpd.GeoDataFrame, color):
        def get_info_text():
            return '<h4>{}</h4>'.format(nuts_ids_column) + '<br>'.join(
                [str(shapes[col].values[0]) for col in info_columns if col in shapes.columns])

        def hover_event_handler(**kwargs):
            info_widget_html.value = get_info_text()
            out = interactive_output(_plot_time_hist_values,
                                     dict(data=fixed(relevant_data), timestamp_column=fixed('_timestamp'),
                                          value_column=fixed(color_column), xlims=fixed(date_limits)))
            time_hist.children = [out]

        def click_event_handler(**kwargs):
            if kwargs['event'] != 'click':
                return
            if tweets_table.placeholder == nuts_id:
                tweets_table.placeholder, tweets_table.value = '', ''
            elif len(relevant_data) > 0:
                header = '<b>{} {}</b>'.format(nuts_ids_column, nuts_id)
                tweets_data = relevant_data[[c for c in table_columns if c in relevant_data.columns]].sort_values(
                    table_columns[0]).dropna(axis='columns', how='all')
                tweets_table.value = header + tweets_data.to_html(na_rep='', index=False)
                tweets_table.layout.margin = '5px'
                tweets_table.placeholder = nuts_id

        nuts_id = shapes[nuts_ids_column].values[0]
        style = {'color': color, 'fillColor': color, 'opacity': 0.5, 'weight': 1.9, 'dashArray': '2',
                 'fillOpacity': 0.2}
        hover_style = {'fillColor': color, 'fillOpacity': 0.5, 'weight': 5}
        layer = ipyleaflet.GeoData(geo_dataframe=shapes, style=style, hover_style=hover_style)
        if full_data is None or tweets_table is None or time_hist is None:
            return layer
        relevant_data = _date_filter(full_data[full_data[nuts_ids_column].str.startswith(nuts_id, na=False)],
                                     date_limits).reset_index()
        if time_hist is not None and info_widget_html is not None:
            layer.on_hover(hover_event_handler)
        if tweets_table is not None:
            layer.on_click(click_event_handler)
        return layer

    def get_layer_group(shapes: gpd.GeoDataFrame, colors, group_name='', sorting_column='LEVL_CODE'):
        if sorting_column in shapes:
            sorting = shapes[sorting_column].argsort()
            shapes = shapes.iloc[sorting]
            colors = colors.iloc[sorting]
        layers = [get_layer(shapes.iloc[[i]], color=color) for i, color in enumerate(colors)]
        return ipyleaflet.LayerGroup(layers=layers, name=group_name)

    def get_colors(values: pd.Series, logarithmic: bool = False, cmap='viridis'):
        get_single = lambda v: get_color(v, logarithmic, cmap, vmin, vmax)
        return values.apply(get_single)

    colors = get_colors(data[color_column], logarithmic=logarithmic, cmap=cmap)
    return get_layer_group(gpd.GeoDataFrame(data), colors=colors, group_name=nuts_ids_column)


def get_color(val: float, logarithmic: bool = False, cmap='viridis', vmin=0, vmax=1):
    norm_class = matplotlib.colors.LogNorm if logarithmic else matplotlib.colors.Normalize
    norm = norm_class(vmin=vmin, vmax=vmax)
    cm = matplotlib.cm.get_cmap(cmap)
    return matplotlib.colors.to_hex(cm(norm(val + 1e-5)))


def merge_df(data, nuts_shapes, nuts_ids_column, color_column, level='all', levels=(0, 1, 2, 3)):
    def aggregate_all(df, **kwargs):
        return pd.concat([aggregate_nuts_level(df, level=level, **kwargs) for level in levels])

    def aggregate_nuts_level(df, level, nuts_ids_column='NUTS_ID', aggregatable_columns=('num_persons',),
                             aggregation=np.sum):
        df = df.dropna(subset=aggregatable_columns)
        df[nuts_ids_column] = df[nuts_ids_column].str.slice(0, level + 2)
        agg_df = aggregation(df.groupby(nuts_ids_column)[aggregatable_columns]).reset_index()
        return agg_df[agg_df[nuts_ids_column].apply(len) == level + 2]

    agg_data = data.dropna(subset=[color_column]).groupby(nuts_ids_column).sum().reset_index()
    if len(levels) > 1:
        if level != 'all':
            agg_data = aggregate_nuts_level(agg_data, level=level, aggregatable_columns=[color_column],
                                            nuts_ids_column=nuts_ids_column)
        else:
            agg_data = aggregate_all(agg_data, aggregatable_columns=[color_column], nuts_ids_column=nuts_ids_column)

    merged_df = pd.merge(agg_data, nuts_shapes, left_on=nuts_ids_column, right_on='NUTS_ID')
    return merged_df.dropna(subset=['NUTS_ID', color_column])


def _plot_time_hist_values(data, timestamp_column, value_column, xlims):
    df = data.dropna(subset=[value_column, timestamp_column])
    if len(df) == 0:
        return
    dates = pd.to_datetime(df[timestamp_column]).apply(pd.Timestamp.date)
    time_hist = df.groupby(dates).sum()[value_column].reset_index()
    time_hist['zero'] = 0
    plt.figure(figsize=(5, 3))
    plt.box(False), plt.xlim(*xlims)
    if len(time_hist) > 1:
        plot = plt.plot(time_hist[timestamp_column], time_hist[value_column], label=value_column,
                        marker='o' if len(time_hist) < 50 else '')
        plt.fill_between(time_hist[timestamp_column], time_hist[value_column], time_hist['zero'], alpha=.4)
    else:
        plot = plt.scatter(time_hist[timestamp_column], time_hist[value_column], label=value_column)
        plt.ylim(0, 1.05 * time_hist[value_column].max())
    plt.legend(), plt.xlabel(''), plt.xticks(rotation=45), plt.tight_layout()
    return plot


def _date_filter(data, date_range, timestamp_column='_timestamp'):
    dates = pd.to_datetime(data[timestamp_column]).apply(pd.Timestamp.date)
    return data[(date_range[0] <= dates) & (dates <= date_range[1])]


def plot_geo_shapes_vis(data, nuts_shapes, nuts_ids_columns, color_column, timestamp_column):
    def plot_cbar(name, logarithmic=False):
        vmin, vmax = app_state['vmin'], app_state['vmax']
        if vmin == vmax or any(pd.isna([vmin, vmax])):
            return
        fig, ax = plt.subplots(figsize=(.3, 10))
        norm = matplotlib.colors.LogNorm(vmin, vmax) if logarithmic else matplotlib.colors.Normalize(vmin, vmax)
        cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=plt.get_cmap(name), norm=norm, orientation='vertical')
        return cbar

    def change_date_range(change):
        data = _date_filter(app_state['data'], change['new'], timestamp_column=timestamp_column).dropna(
            subset=nuts_ids_columns, how='all')
        merged_dfs = [merge_df(data=data, nuts_shapes=nuts_shapes, nuts_ids_column=nuts_ids_column,
                               color_column=color_column, level=level_selector.value, levels=app_state['nuts_levels'])
                      for nuts_ids_column in nuts_ids_columns]
        app_state['vmax'] = np.max([df[color_column].max() for df in merged_dfs])
        interactive_output(plot_cbar, dict(name=cmap_selector, logarithmic=logarithmic_cbox))

        m.layers = [l for l in m.layers if type(l) != ipyleaflet.LayerGroup]
        for merged_df, nuts_ids_column in zip(merged_dfs, nuts_ids_columns):
            table_columns = ['_timestamp', 'text_translated', 'num_persons', 'mode',
                             *[col for col in nuts_ids_columns if col != nuts_ids_column]]
            layer = get_shapes_heatmap(data=merged_df, nuts_ids_column=nuts_ids_column, color_column=color_column,
                                       info_widget_html=info_widget, vmin=app_state['vmin'], vmax=app_state['vmax'],
                                       full_data=app_state['full_data'], time_hist=time_hist, date_limits=change['new'],
                                       tweets_table=tweets_table, cmap=app_state['cmap'], table_columns=table_columns,
                                       logarithmic=app_state['logarithmic'])
            m.add_layer(layer)

        if 'full_groups' not in app_state:
            app_state['full_groups'] = [l.layers for l in m.layers if type(l) is ipyleaflet.LayerGroup]

        cbar_widget.children = [interactive_output(plot_cbar, dict(name=cmap_selector, logarithmic=logarithmic_cbox))]

    def change_level_layers(change={}):
        def change_layers(layer_group: ipyleaflet.LayerGroup, all_layers: list):
            new_layers = [l for l in all_layers if
                          l.data['features'][0]['properties']['LEVL_CODE'] == app_state['level']]
            layer_group.layers = new_layers if app_state['level'] != 'all' else all_layers

        layer_groups = [l for l in m.layers if type(l) is ipyleaflet.LayerGroup]
        if 'new' in change:
            app_state['level'] = change['new']
        for layer_group, full_group in zip(layer_groups, app_state['full_groups']):
            change_layers(layer_group, full_group)

    def change_colormap():
        cmap, logarithmic = app_state['cmap'], app_state['logarithmic']

        def update_layer(l):
            val = l.data['features'][0]['properties'][color_column]
            color = get_color(val, cmap=cmap, vmin=app_state['vmin'], vmax=app_state['vmax'], logarithmic=logarithmic)
            l.style.update({'fillColor': color, 'color': color})
            l.hover_style.update({'fillColor': color, 'color': color})
            new_layer = ipyleaflet.GeoJSON(data=l.data, style=l.style, hover_style=l.hover_style)
            new_layer._hover_callbacks = l._hover_callbacks
            new_layer._click_callbacks = l._click_callbacks
            return new_layer

        app_state['full_groups'] = [[update_layer(l) for l in layers] for layers in app_state['full_groups']]

        change_level_layers()

    def change_colormap_name(change):
        app_state['cmap'] = change['new']
        change_colormap()

    def change_colormap_log(change):
        app_state['logarithmic'] = change['new']
        change_colormap()

    def add_widget(widget, pos, margin='0px 0px 0px 0px'):
        widget.layout.margin = margin
        widget_control = ipyleaflet.WidgetControl(widget=widget, position=pos)
        m.add_control(widget_control)

    def on_zoom(change, offset=-5):
        if m.zoom != app_state['zoom']:
            app_state['zoom'] = m.zoom
            if level_on_zoom.value:
                min_level, max_level = np.min(app_state['nuts_levels']), np.max(app_state['nuts_levels'])
                level = min(max_level, max(min_level, m.zoom + offset))
                app_state['level'] = level
                level_selector.value = level

    app_state = dict(zoom=4, data=data.dropna(subset=nuts_ids_columns, how='all'), cmap='viridis', logarithmic=False,
                     vmin=1, vmax=1, full_data=data.copy(), level='all',
                     nuts_levels=sorted(nuts_shapes['LEVL_CODE'].unique()))
    m = ipyleaflet.Map(center=(51, 10), zoom=app_state['zoom'], scroll_wheel_zoom=True, zoom_control=False)
    m.layout.height = '900px'

    info_widget_default_text = 'Hover over a Region<br>Click it to see tweets'
    info_widget = widgets.HTML(info_widget_default_text)
    add_widget(info_widget, pos='topright', margin='0px 5px 0px 5px')

    loading_wrapper = _loading_wrapper_factory(info_widget, info_widget_default_text)

    time_hist = widgets.HBox([])
    add_widget(time_hist, pos='bottomleft')

    tweets_table = widgets.HTML(layout=widgets.Layout(overflow='scroll_hidden'))
    tweets_box = widgets.HBox([tweets_table], layout=Layout(max_height='400px', overflow_y='auto', max_width='900px'))
    add_widget(tweets_box, pos='bottomleft')

    time_slider = get_time_slider(app_state['data'])
    time_slider.observe(loading_wrapper(change_date_range), type='change', names=('value',))
    add_widget(time_slider, 'topleft', margin='0px 5px 0px 5px')

    level_selector = widgets.Dropdown(options=['all', *app_state['nuts_levels']], description='NUTS levels',
                                      layout=Layout(max_width='180px'))
    level_selector.observe(handler=loading_wrapper(change_level_layers), type='change', names=('value',))
    level_on_zoom = widgets.Checkbox(value=True, description='with zoom', layout=Layout(max_width='180px'))
    level_control = widgets.VBox([level_selector, level_on_zoom])

    cmap_selector = widgets.Dropdown(options=['viridis', 'inferno', 'magma', 'winter', 'cool'], description='colormap',
                                     layout=Layout(max_width='180px'))
    logarithmic_cbox = widgets.Checkbox(description='logarithmic', layout=Layout(max_width='180px'))
    cmap_control = widgets.VBox([cmap_selector, logarithmic_cbox])
    cmap_selector.observe(handler=loading_wrapper(change_colormap_name), type='change', names=('value',))
    logarithmic_cbox.observe(handler=loading_wrapper(change_colormap_log), type='change', names=('value',))
    add_widget(widgets.HBox([level_control, cmap_control]), pos='topleft', margin='5px 5px 0px 5px')

    cbar_widget = widgets.HBox([interactive_output(plot_cbar, dict(name=cmap_selector, logarithmic=logarithmic_cbox))])
    add_widget(cbar_widget, pos='bottomright')

    m.add_control(ipyleaflet.LayersControl())
    m.add_control(ipyleaflet.FullScreenControl())
    m.observe(handler=on_zoom)

    change_date_range(dict(new=time_slider.value))
    display(m)


def geo_vis_shapes_app(data: pd.DataFrame, simplify_nuts_shapes=True, nuts_shapes: gpd.GeoDataFrame = None,
                       nuts_ids_columns=('origin', 'destination'), timestamp_column='_timestamp',
                       color_column='num_persons', shp_folder='nuts_data'):
    if nuts_shapes is None:
        nuts_shapes = get_nuts_shapes(shp_folder=shp_folder, simplify=simplify_nuts_shapes, tol=1e-3)

    geo_vis = interactive_output(plot_geo_shapes_vis,
                                 dict(nuts_ids_columns=fixed(nuts_ids_columns), data=fixed(data),
                                      timestamp_column=fixed(timestamp_column), nuts_shapes=fixed(nuts_shapes),
                                      color_column=fixed(color_column)))
    display(geo_vis)


def _wkb_hex_to_point(s):
    return list(shapely.wkb.loads(s, hex=True).coords)[0][::-1]


def _get_timestap_column(df):
    return [c for c in df.columns if 'date' in c or 'timestamp' in c][0]


def _to_html(val):
    if type(val) in (list, tuple, np.array):
        return ', '.join(list(map(_to_html, val)))
    if type(val) is str:
        if val.startswith('http'):
            disp = val
            if val.endswith('png') or val.endswith('.jpg'):
                disp = '<img src="{}" width="250px" style="padding:3px">'.format(val, val)
            return '<a href={} target="_blank">{}</a>'.format(val, disp)
    return str(val)


def _loading_wrapper_factory(info_box, info_box_default_value=''):
    def loading_wrapper(function):
        def loading_func(*args, **kwargs):
            prev_layout = info_box.layout
            info_box.value, info_box.layout = '<b>loading...</b>', Layout(margin='5px 15px 5px 15px')
            function(*args, **kwargs)
            info_box.value, info_box.layout = info_box_default_value, prev_layout

        return loading_func

    return loading_wrapper


def get_marker_cluster(data, geom_column, info_box: widgets.HTML, timestamp_column, title_columns=()):
    def get_title(d):
        return '<br>'.join([_to_html(d[c]) for c in title_columns if d[c] not in (np.nan, None)])

    def get_hover_event_handler(info):
        def hover_event_handler(**kwargs):
            info_box.value = info

        return hover_event_handler

    locs = data[geom_column].apply(_wkb_hex_to_point)
    dicts = data.to_dict(orient='rows')

    markers = [ipyleaflet.Marker(location=loc, title=str(loc), draggable=False) for loc in locs]
    clusters = ipyleaflet.MarkerCluster(markers=markers, name='Marker Cluster')
    for marker, d in zip(clusters.markers, dicts):
        marker.on_mouseover(get_hover_event_handler(get_title(d)))
        marker.timestamp = pd.to_datetime(d[timestamp_column])
    return clusters


def _plot_time_hist_counts(full_data, filtered_data, timestamp_column):
    def plot_counts(df, color):
        dates = pd.to_datetime(df[timestamp_column]).apply(pd.Timestamp.date)
        counts = df.groupby(dates)[timestamp_column].count()
        plot = plt.plot(counts, c=color, alpha=.7)
        plt.fill_between(counts.index, counts, counts * 0, alpha=.4, color=color)
        return plot

    plt.figure(figsize=(6.8, .7))
    plt.box(False), plt.axis('off')
    plot_counts(full_data, '#1f77b4')
    return plot_counts(filtered_data, '#d62728')


def _get_ts_count_plot(full_data, filtered_data, timestamp_column):
    ts_plot = interactive_output(_plot_time_hist_counts,
                                 dict(full_data=fixed(full_data), filtered_data=fixed(filtered_data),
                                      timestamp_column=fixed(timestamp_column)))
    ts_plot.layout.margin = '-10px 0px -15px -17px'
    return ts_plot


def plot_geo_data_cluster(data, geom_column, title_columns):
    def date_filter(df, date_range, timestamp_column: str):
        dates = pd.to_datetime(df[timestamp_column]).apply(pd.Timestamp.date)
        return df[(date_range[0] <= dates) & (dates <= date_range[1])]

    def toggle_tweets_table_visibility(change):
        app_state['is_in_bounds'] = None
        if change['old'] and not change['new']:
            tweets_table.value, app_state['is_in_bounds'] = '', None
        else:
            update_range()

    def update_range(*_):
        def in_bounds(loc):
            return all(bounds[0] < loc) and all(loc < bounds[1])

        if len(m.bounds) == 0:
            return

        bounds = np.array(m.bounds)
        locs = app_state['filtered_data'][geom_column].apply(_wkb_hex_to_point)
        is_in_bounds = locs.apply(in_bounds)
        if app_state['is_in_bounds'] is None or not np.array_equal(is_in_bounds, app_state['is_in_bounds']):
            if is_in_bounds.sum() > 0:
                ts_plot = _get_ts_count_plot(app_state['full_data'], app_state['filtered_data'][is_in_bounds],
                                             app_state['timestamp_column'])
                time_slider_box.children = [ts_plot, *time_slider_box.children[1:]]

                if tweets_table_cb.value:
                    tweets_table.value = app_state['filtered_data'].loc[is_in_bounds, title_columns].reset_index(
                        drop=True).to_html(formatters={'media': _to_html}, escape=False, na_rep='', index=False)
            else:
                tweets_table.value = ''

        tweets_table_cb.description = 'Show {} Tweets'.format(is_in_bounds.sum())
        app_state['is_in_bounds'] = is_in_bounds

    def change_date_range(change):
        def filter_markers(marker_cluster: ipyleaflet.MarkerCluster, min_date, max_date):
            marker_cluster.markers = [m for m in app_state['markers'] if min_date <= m.timestamp.date() <= max_date]

        filtered_data = date_filter(app_state['full_data'], change['new'],
                                    timestamp_column=app_state['timestamp_column'])
        app_state['filtered_data'] = filtered_data
        heatmap.locations = list(filtered_data[geom_column].apply(_wkb_hex_to_point).values)
        filter_markers(marker_clusters, *change['new'])
        update_range()

    m = ipyleaflet.Map(center=(51, 10), zoom=4, scroll_wheel_zoom=True, zoom_control=False)
    m.layout.height = '900px'

    app_state = dict(is_in_bounds=None, full_data=data.dropna(subset=[geom_column]),
                     filtered_data=data.dropna(subset=[geom_column]), timestamp_column=_get_timestap_column(data))

    info_box_default_text = 'Hover over a marker'
    info_box = widgets.HTML(info_box_default_text,
                            layout=Layout(margin='10px', overflow='scroll_hidden'))
    m.add_control(ipyleaflet.WidgetControl(widget=widgets.HBox([info_box], layout=Layout(max_height='300px')),
                                           position='topright'))

    loading_wrapper = _loading_wrapper_factory(info_box, info_box_default_text)

    tweets_table = widgets.HTML(layout=widgets.Layout(overflow='scroll_hidden'))
    tweets_table_box = widgets.HBox([tweets_table],
                                    layout=Layout(max_height='500px', overflow_y='auto', max_width='900px'))
    tweets_table_cb = widgets.Checkbox(value=False)
    tweets_table_cb.observe(toggle_tweets_table_visibility, type='change', names=('value',))
    tweets_box = widgets.VBox([tweets_table_cb, tweets_table_box])
    m.add_control(ipyleaflet.WidgetControl(widget=tweets_box, position='bottomleft'))

    time_slider = get_time_slider(app_state['full_data'])
    time_slider.observe(loading_wrapper(change_date_range), type='change', names=('value',))
    time_slider.layout.margin, time_slider.description = '0px 5px 0px 5px', ''
    ts_plot = _get_ts_count_plot(data, app_state['filtered_data'], app_state['timestamp_column'])
    time_slider_box = widgets.VBox([ts_plot, time_slider])
    m.add_control(ipyleaflet.WidgetControl(widget=time_slider_box, position='topleft'))

    marker_clusters = get_marker_cluster(app_state['filtered_data'], geom_column, title_columns=title_columns,
                                         info_box=info_box, timestamp_column=app_state['timestamp_column'])
    app_state['markers'] = marker_clusters.markers

    heatmap_locations = list(app_state['filtered_data'][geom_column].apply(_wkb_hex_to_point).values)
    heatmap = ipyleaflet.Heatmap(name='Heatmap', min_opacity=.1, blur=20, radius=20, max_zoom=12,
                                 locations=heatmap_locations)
    m.add_layer(heatmap)
    m.add_layer(marker_clusters)

    change_date_range(dict(new=time_slider.value))
    m.add_control(ipyleaflet.LayersControl())
    m.add_control(ipyleaflet.FullScreenControl())
    m.observe(update_range)

    display(m)


def geo_vis_cluster_app(data, geom_column='geom_tweet'):
    title_columns = widgets.SelectMultiple(options=sorted(data.columns), description='Information to show',
                                           value=('text', 'text_translated', '_timestamp', 'media'))
    title_columns_tip = widgets.HTML('Select multiple by dragging or ctrl + click <br> Deselect with ctrl + click')
    title_columns_controls = widgets.HBox([title_columns, title_columns_tip])

    geo_vis = interactive_output(plot_geo_data_cluster,
                                 dict(data=fixed(data), geom_column=fixed(geom_column), title_columns=title_columns))

    geo_vis.layout.width = '90%'
    controls = widgets.Tab([title_columns_controls])
    controls.set_title(0, 'Information to Show')
    display(controls, geo_vis)
