import pandas as pd

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Div
from bokeh.io import curdoc
from bokeh.models.widgets import Select
from bokeh.layouts import layout

cand_file = "20170424194451.ak10.cand" 

cands = pd.read_csv(cand_file, header=0, delim_whitespace=True)

# rename first column from #snr to snr
#col_names = cands.columnes.values
#col_names[0] = 'snr'
#cands.columns = col_names

axis_map = {
    "S/N": "#snr",
    "Sample No.": "sample",
    "Time (s)": "time",
    "log2(Boxcar width)": "filter",
    "DM trial": "dm_trial",
    "DM": "dm",
    "Member count": "members",
    "Begin (?)": "begin",
    "End (?)": "end",
    "Beam No.": "beam",
    "Antenna": "antenna",
}

x_axis = Select(title="X Axis", options=sorted(axis_map.keys()), value="DM")
y_axis = Select(title="Y Axis", options=sorted(axis_map.keys()), value="Time (s)")

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data=dict(x=[], y=[], DM=[], snr=[], filter_width=[]))#, alpha=[]))

p = figure(plot_height=600, plot_width=700, title="", tools = 'box_zoom,reset' )#, tools=[hover])
p.circle(x="x", y="y", source=source, size=7)#, color="color", line_color=None)#, fill_alpha="alpha")

def select_cands():
    return cands

def update():
    df = select_cands()
    x_name = axis_map[x_axis.value]
    y_name = axis_map[y_axis.value]

    p.xaxis.axis_label = x_axis.value
    p.yaxis.axis_label = y_axis.value
    p.title.text = "%d candidates selected" % len(df)
    print "yooo"
    source.data = dict(
        x=df[x_name],
        y=df[y_name],
        DM=df["dm"],
        snr=df["#snr"],
        filter_width=df["filter"],
    )
        #color=df["color"],
        #alpha=df["alpha"],

update()  # initial load of the data


desc = Div()
l = layout([[desc], [p],])
curdoc().add_root(l)
curdoc().title = "Candidates"
