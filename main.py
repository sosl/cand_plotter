import ConfigParser
import os, glob

import pandas as pd

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Div
from bokeh.io import curdoc
from bokeh.models.widgets import Select
from bokeh.layouts import layout, widgetbox

# for fbank handling
import numpy as np
import mbplotlib
from sigpyproc.Readers import FilReader
CAND_PLOT_CFG = os.environ['HOME']+"/.candplotter.cfg"

config = ConfigParser.ConfigParser()
config.read(CAND_PLOT_CFG)
if not config.sections():
    raise RuntimeError("Config file not found", CAND_PLOT_CFG)

TOP_DIR = config.get('data', 'topdir')
CAND_TOP_DIR = config.get('cand', 'topdir')

SB_ids = [os.path.basename(SB) for SB in glob.glob(TOP_DIR+"/SB*")]
SB_selector = Select(title="Scheduling block", options=sorted(SB_ids))#, value=SB_ids[0])

UTCs = [os.path.basename(UTC) for UTC in glob.glob(TOP_DIR+SB_selector.value+"/20*")]
UTC_selector = Select(title="UTC", options=sorted(UTCs))#, value=UTCs[0])


antennas = [os.path.basename(cf) for cf in glob.glob(TOP_DIR+SB_selector.value+"/" + UTC_selector.value + "/ak*")]
antennas_selector = Select(title="Antenna file", options = sorted(antennas))#, value= antennas[0])

cand_file = CAND_TOP_DIR + SB_selector.value + "/" + UTC_selector.value + "." + antennas_selector.value + ".cand"

def update_SB():
    UTCs = [os.path.basename(UTC) for UTC in glob.glob(TOP_DIR+SB_selector.value+"/20*")]
    print "found", len(UTCs), "UTCs"
    UTC_selector.options=sorted(UTCs)

    antennas = [os.path.basename(cf) for cf in glob.glob(TOP_DIR+SB_selector.value+"/" + UTC_selector.value + "/ak*")]
    antennas_selector.options = sorted(antennas)

def update_UTC():
    antennas = [os.path.basename(cf) for cf in glob.glob(TOP_DIR+SB_selector.value+"/" + UTC_selector.value + "/ak*")]
    antennas_selector.options = sorted(antennas)
    print "update_UTC", CAND_TOP_DIR+SB_selector.value+"/" + UTC_selector.value

cands = pd.DataFrame()
def update_cand_file():
    cand_file = CAND_TOP_DIR + SB_selector.value + "/" + UTC_selector.value + "." + antennas_selector.value + ".cand"
    print "loading cands", cand_file
    _cands = pd.read_csv(cand_file, header=0, delim_whitespace=True)
    for column in _cands.columns.values:
        cands[column] = _cands[column]
    print "loaded cands", _cands["beam"].count(), cands["beam"].count()
    update()

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

cand_x_axis = Select(title="X Axis", options=sorted(axis_map.keys()), value="DM")
cand_y_axis = Select(title="Y Axis", options=sorted(axis_map.keys()), value="Time (s)")

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data=dict(x=[], y=[], DM=[], snr=[], filter_width=[],
    sample=[], beam=[], color=[]))#, alpha=[]))

source_ts = ColumnDataSource(data=dict(time=[], series=[]))
source_fb = ColumnDataSource(data=dict(image=[]))
source_fb_conv = ColumnDataSource(data=dict(image=[]))

TOOLS = 'crosshair, box_zoom, reset, box_select, tap'

cands_fig = figure(plot_height=600, plot_width=700, title="", tools = TOOLS,
        toolbar_location='right')
cands_plot = cands_fig.circle(x="x", y="y", source=source, size=7, color="color", line_color=None)#, fill_alpha="alpha")

timeseries_fig = figure(plot_height=300, plot_width=1400, title="Time Series",
        tools = 'box_zoom, reset', toolbar_location='right')
timeseries_plot = timeseries_fig.line(x="time", y="series", source=source_ts, line_width=2)

dedisp_fig = figure(plot_height=600, plot_width=700, title="Dedispersed data",
        tools ='box_zoom, reset', x_range=(0, 10), y_range=(0, 10),
        toolbar_location='right' )
dedisp_plot = dedisp_fig.image(image="image", x=0, y=0, dw=10, dh=10, source=source_fb, palette = 'Viridis256' )

conv_fig = figure(plot_height=600, plot_width=700, title="Convolved data",
        tools ='box_zoom, reset', x_range=(0, 10), y_range=(0, 10),
        toolbar_location='right')
conv_plot = conv_fig.image(image="image", x=0, y=0, dw=10, dh=10,
        source=source_fb_conv, palette = 'Viridis256')

def select_cands():
    cands["color"] = pd.Series("red", cands.index)
    return cands

def update():
    df = select_cands()
    x_name = axis_map[cand_x_axis.value]
    y_name = axis_map[cand_y_axis.value]

    cands_fig.xaxis.axis_label = cand_x_axis.value
    cands_fig.yaxis.axis_label = cand_y_axis.value
    cands_fig.title.text = "%d candidates present" % len(df)
    print "yooo"
    source.data = dict(
        x=df[x_name],
        y=df[y_name],
        DM=df["dm"],
        snr=df["#snr"],
        filter_width=df["filter"],
        color=df["color"],
        sample=df["sample"],
        beam=df["beam"],
    )
        #alpha=df["alpha"],

def tap_callback(attr, old, new):
    if len(new['1d']['indices']) > 0:
        cand_id = new['1d']['indices'][0]
        dm = source.data["DM"][cand_id]
        sample = source.data["sample"][cand_id]
        filter_ind = source.data["filter_width"][cand_id]
        beam = source.data["beam"][cand_id]
        _, _dedisp_block, _conv_block, _time, _series = get_fbank_data(dm,
                sample, 2**filter_ind, beam)
        source_ts.data["time"] = _time
        source_ts.data["series"] = _series
        source_fb.data["image"] = [_dedisp_block]
        source_fb_conv.data["image"] = [_conv_block]

def get_fbank_data(dm, sample, width, beam):
    # based on Wael's filplot
    fil_fn = glob.glob(TOP_DIR+"/" + SB_selector.value+"/" + UTC_selector.value
            + "/" + antennas_selector.value + "/C000/*."+ "%02d" % beam +".fil")[0]
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

    return block, disp_block[:,:event_end], conv_arr, time, series

cands_plot.data_source.on_change('selected', tap_callback)

top_level_controls = [ SB_selector, UTC_selector, antennas_selector]
SB_selector.on_change('value', lambda attr, old, new: update_SB())
UTC_selector.on_change('value', lambda attr, old, new: update_UTC())
antennas_selector.on_change('value', lambda attr, old, new: update_cand_file())

cand_controls = [cand_x_axis, cand_y_axis]
for control in cand_controls:
    control.on_change('value', lambda attr, old, new: update())

sizing_mode = 'fixed'
top_level_inputs = widgetbox(*top_level_controls, sizing_mode=sizing_mode)
cand_control_inputs = widgetbox(*cand_controls, sizing_mode=sizing_mode)

desc = Div()
l = layout([[desc], [top_level_inputs], [cands_fig, cand_control_inputs],
    [timeseries_fig], [dedisp_fig, conv_fig]])
curdoc().add_root(l)
curdoc().title = "Candidates"
