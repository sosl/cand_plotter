import pandas as pd

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Div
from bokeh.io import curdoc
from bokeh.models.widgets import Select
from bokeh.layouts import layout

# for fbank handling
import numpy as np
import mbplotlib
from sigpyproc.Readers import FilReader

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
source = ColumnDataSource(data=dict(x=[], y=[], DM=[], snr=[], filter_width=[],
    color=[]))#, alpha=[]))

time = []
series = []
source_ts = ColumnDataSource(data=dict(time=time, series=series))
source_fb = ColumnDataSource(data=dict(image=[]))

TOOLS = 'crosshair, box_zoom, reset, box_select, tap'

cands_fig = figure(plot_height=600, plot_width=700, title="", tools = TOOLS )#, tools=[hover])
cands_plot = cands_fig.circle(x="x", y="y", source=source, size=7, color="color", line_color=None)#, fill_alpha="alpha")

timeseries_fig = figure(plot_height=600, plot_width=700, title="Time Series", tools = TOOLS )
timeseries_plot = timeseries_fig.line(x="time", y="series", source=source_ts, line_width=2)

dedisp_fig = figure(plot_height=600, plot_width=700, title="dedisp", tools ='box_zoom, reset', x_range=(0, 10), y_range=(0, 10) )
dedisp_plot = dedisp_fig.image(image="image", x=0, y=0, dw=10, dh=10, source=source_fb, palette = 'Viridis256' )
def select_cands():
    cands["color"] = pd.Series("red", cands.index)
    return cands

def update():
    df = select_cands()
    x_name = axis_map[x_axis.value]
    y_name = axis_map[y_axis.value]

    cands_fig.xaxis.axis_label = x_axis.value
    cands_fig.yaxis.axis_label = y_axis.value
    cands_fig.title.text = "%d candidates present" % len(df)
    print "yooo"
    source.data = dict(
        x=df[x_name],
        y=df[y_name],
        DM=df["dm"],
        snr=df["#snr"],
        filter_width=df["filter"],
        color=df["color"],
    )
        #alpha=df["alpha"],

def tap_callback(attr, old, new):
    print "Selected candidate with index", new['1d']['indices'][0]
    _, _dedisp_block, _, time, series = get_fbank_data(478.399, 50686, 2**3)
    source_ts.data["time"] = time
    source_ts.data["series"] = series
    source_fb.data["image"] = [_dedisp_block]

def get_fbank_data(dm, sample, width):
    # based on Wael's filplot
    fil_fn = "2017-04-24-19:45:44_0000000000000000.000000.21.fil"
    fil = FilReader(fil_fn)

    tsamp = fil.header.tsamp
    tsamp_ms = fil.header.tsamp*1000.
    backstep = int(200/tsamp_ms)
    event_end = int(backstep*2 + width)

    bw = fil.header.bandwidth

    t_smear = np.ceil(((fil.header.bandwidth*8.3*dm)
            / (fil.header.fcenter*10**(-3))**3)/(tsamp*1000000))
    t_smear = int(1.05*t_smear)
    t_extract = 2*backstep + 2*width + t_smear

    if (sample-backstep+t_extract) > fil.header.nsamples:
            raise RuntimeError("Filterbank out-of-bound", "End window is out of bounds")
    # original filterbank
    block = fil.readBlock(sample-backstep, t_extract)
    # dedisperse d filterbank:
    disp_block = block.dedisperse(dm)

    # dedispersed filterbank convolved at the expected width
    conv_arr = np.zeros((block.shape[0],event_end))

    for i in xrange(conv_arr.shape[0]):
            conv_arr[i] = mbplotlib.wrapper_conv_boxcar(np.array(disp_block[i,:event_end],
                dtype=np.ctypeslib.ct.c_long),width)
    conv_arr = conv_arr[:,:(-width-1)]

    time  = np.arange(event_end)*tsamp_ms
    series = disp_block.sum(axis=0)[:event_end]

    return block, disp_block, conv_arr, time, series

cands_plot.data_source.on_change('selected', tap_callback)

update()  # initial load of the data


desc = Div()
l = layout([[desc], [cands_fig], [timeseries_fig, dedisp_fig],])
curdoc().add_root(l)
curdoc().title = "Candidates"
