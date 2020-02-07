import ipywidgets as widgets
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from ipywidgets import fixed, Layout, interactive_output
from matplotlib import pyplot as plt


def paginate_df(df, nb_items=10):
    def show_df(df, page):
        display(df.head(page * nb_items).tail(nb_items))

    def show_next(change):
        pagination_slider.value += change

    def get_pagination_buttons():
        next_button = widgets.Button(layout=widgets.Layout(width='30px'), icon='chevron-right')
        next_button.on_click(lambda _: show_next(1))
        prev_button = widgets.Button(layout=widgets.Layout(width='30px'), icon='chevron-left')
        prev_button.on_click(lambda _: show_next(-1))
        return [prev_button, next_button]

    nb_rows = widgets.Label(value='{} rows'.format(len(df)))

    if len(df) > nb_items:
        nb_pages = int(np.ceil(len(df) / nb_items))
        pagination_slider = widgets.IntSlider(value=1, min=1, max=nb_pages, layout=Layout(width='60%'))
        pagination_slider_label = widgets.Label(value='of {} pages with '.format(nb_pages))

        pagination_controls = widgets.HBox(
            [pagination_slider, pagination_slider_label, nb_rows, *get_pagination_buttons()])
        paginated_table = interactive_output(show_df, dict(df=fixed(df), page=pagination_slider))
        display(pagination_controls, paginated_table)
    else:
        display(nb_rows, df)


def plot_column_dist(df: pd.DataFrame, column: str, nb_to_plot=10):
    def is_text(vals):
        if type(vals.dropna().values[0]) != str:
            return False

        mean_nb_tokens = vals.dropna().str.split(' ').apply(len).mean()
        return mean_nb_tokens > 3

    def is_numeric(vals):
        return all(vals.apply(type).unique() == float) and len(vals.unique()) > nb_to_plot and not any(vals.isna())

    vals = df[column]
    if len(vals.dropna()) == 0:
        plt.title('No values to plot'), plt.show()
        return

    if is_numeric(vals):
        sns.distplot(vals)
    else:
        if type(vals.dropna().values[0]) in (list, tuple, set) or is_text(vals):
            if is_text(vals):
                vals = vals.str.split(' ')
                vals.name = 'tokens'
            concat_series = pd.Series(np.concatenate(vals.dropna().values))
            counts = pd.DataFrame(concat_series.value_counts())
        else:
            counts = pd.DataFrame(vals.value_counts())
        counts.head(nb_to_plot).plot(kind='bar', label=vals.name)
    plt.tight_layout(), plt.grid(), plt.title(str(nb_to_plot) + ' most frequent ' + vals.name), plt.show()


def filter_df(df, order_by='uniqueid', ascending=True, required=(), search_column=None, search_term=''):
    df = df.sort_values(order_by, ascending=ascending)

    if len(required) > 0:
        df = df.dropna(subset=required)

    if search_term not in (None, '') and search_column in df.keys():
        df = df[df[search_column].astype(str).str.lower().str.contains(search_term.lower())]

    return df


def show_filtered_df(df, order_by='uniqueid', ascending=True, nb_items=10, required=(), search_column=None,
                     search_term=''):
    df = filter_df(df, order_by=order_by, ascending=ascending, required=required, search_column=search_column,
                   search_term=search_term)
    paginate_df(df, nb_items=nb_items)


def table_app(df):
    def save_filtered_data(_):
        filtered_df = filter_df(df, order_by=order_by.value, ascending=ascending.value, required=filter_selector.value,
                                search_column=search_column.value, search_term=search_term.value)
        filtered_df.to_csv(save_fn.value)
        save_status.value = '\t  sucessfully saved {} rows as {}.'.format(len(filtered_df), save_fn.value)

    def plot_filtered(df, required, search_column, search_term, plot_column):
        filtered_df = filter_df(df, order_by=order_by.value, required=required, search_column=search_column,
                                search_term=search_term)
        plot_column_dist(df=filtered_df, column=plot_column)

    nb_items = widgets.Dropdown(options=[10, 20, 50], description='items per page', layout=Layout(width='20%'))
    order_by = widgets.Dropdown(options=sorted(df.keys()), description='order by')
    ascending = widgets.ToggleButton(value=True, description='ascending')
    sorting = widgets.HBox([order_by, ascending, nb_items], layout=Layout(height='50px'))

    filter_selector = widgets.SelectMultiple(options=sorted(df.keys()))

    filter_tip = widgets.VBox([widgets.HTML('Select multiple by dragging or ctrl + click'),
                               widgets.HTML('Deselect with ctrl + click')])
    filtering = widgets.HBox([filter_selector, filter_tip])

    save_button = widgets.Button(description='save')
    save_fn = widgets.Text('filtered_data.csv')
    save_button.on_click(save_filtered_data)
    save_status = widgets.Label()
    saving = widgets.HBox([save_fn, save_button, save_status])

    search_term = widgets.Text('', tooltip='Search')
    search_column = widgets.Dropdown(options=df.keys())
    plot_column = widgets.Dropdown(options=df.keys())

    column_dist = interactive_output(plot_filtered,
                                     dict(df=fixed(df), search_column=search_column, required=filter_selector,
                                          search_term=search_term, plot_column=plot_column))
    column_plot_box = widgets.VBox([widgets.Label('Plot Columns'), plot_column, column_dist])
    search_box = widgets.VBox([widgets.Label('Search Columns'), search_column, search_term])
    searching = widgets.TwoByTwoLayout(top_left=search_box, top_right=column_plot_box)
    widgets.dlink((search_column, 'value'), (plot_column, 'value'))

    accordion = widgets.Tab(children=[sorting, filtering, searching, saving])
    accordion.set_title(0, 'Sorting')
    accordion.set_title(1, 'Required Values')
    accordion.set_title(2, 'Searching')
    accordion.set_title(3, 'Save filtered Data')

    interactive_table = interactive_output(show_filtered_df,
                                           dict(df=fixed(df), order_by=order_by, nb_items=nb_items,
                                                required=filter_selector,
                                                ascending=ascending, search_column=search_column,
                                                search_term=search_term))
    display(widgets.VBox([accordion, interactive_table]))
