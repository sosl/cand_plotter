import ConfigParser
import os, glob

import pandas as pd

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Div
from bokeh.io import curdoc
from bokeh.models.widgets import Select, Slider
from bokeh.layouts import layout, widgetbox
#choose palette for filter to color mapping. If more than 8 filter widths allowed, different palette will be needed
from bokeh.palettes import Spectral8

filter_palette = Spectral8

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
    _cands = pd.read_csv(cand_file, header=None, comment='#',
            delim_whitespace=True,
            names=config.get('cand', 'format').split(',') )
    _cands["color"] = pd.Series("blue", _cands.index)
    # set the range of threshold sliders:
    cand_min_snr.start=_cands["snr"].min()
    cand_min_snr.end=_cands["snr"].max()
    cand_min_width.start=_cands["logwidth"].min()
    cand_min_width.end=_cands["logwidth"].max()
    cand_max_width.start=_cands["logwidth"].min()
    cand_max_width.end=_cands["logwidth"].max()
    cand_min_DM.start=_cands["DM"].min()
    cand_min_DM.end=_cands["DM"].max()
    cand_max_DM.start=_cands["DM"].min()
    cand_max_DM.end=_cands["DM"].max()
    for column in _cands.columns.values:
        cands[column] = _cands[column]
    update()

axis_map = {
    "S/N": "snr",
    "Sample No.": "sample",
    "Time (s)": "time",
    "log2(Boxcar width)": "logwidth",
    "DM trial": "dm_trial",
    "DM": "DM",
    "Member count": "members",
    "Begin (?)": "begin",
    "End (?)": "end",
    "Beam No.": "beam",
    "Antenna": "antenna",
}

cand_x_axis = Select(title="X Axis", options=sorted(axis_map.keys()),
        value="Time (s)")
cand_y_axis = Select(title="Y Axis", options=sorted(axis_map.keys()),
        value="DM")

cand_min_snr = Slider(title="Min S/N", value = 6.0, start=6.0, end=15.0, step=0.5)
cand_min_width = Slider(title="Min log2(width)", value = 0, start=0, end=8, step=1)
cand_max_width = Slider(title="Max log2(width)", value = 8, start=0, end=8, step=1)
cand_min_DM = Slider(title="Min DM", value = 0., start=0., end=4116., step=1)
cand_max_DM = Slider(title="Max DM", value = 4116., start=0., end=4116., step=1)

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data=dict(x=[], y=[], DM=[], snr=[], filter_width=[],
    sample=[], beam=[], color=[], alpha=[]))

source_ts = ColumnDataSource(data=dict(time=[], series=[]))
source_fb = ColumnDataSource(data=dict(image=[]))
source_fb_conv = ColumnDataSource(data=dict(image=[]))

TOOLS = 'crosshair, box_zoom, reset, box_select, tap'

cands_fig = figure(plot_height=600, plot_width=700, title="", tools = TOOLS,
        toolbar_location='right')
cands_plot = cands_fig.circle(x="x", y="y", source=source, size=15,
        color="color", line_color=None, fill_alpha="alpha")
cands_fig.text(x="x", y="y", text="beam", source=source, text_font_size='8pt',
        x_offset=-5, y_offset=5)

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
    selected = cands[
            (cands.snr >= cand_min_snr.value) &
            (cands.logwidth >= cand_min_width.value) &
            (cands.logwidth <= cand_max_width.value) &
            (cands.DM >= cand_min_DM.value) &
            (cands.DM <= cand_max_DM.value)
    ]
    print "selected", len(selected)
    return selected

def update():
    df = select_cands()
    x_name = axis_map[cand_x_axis.value]
    y_name = axis_map[cand_y_axis.value]

    cands_fig.xaxis.axis_label = cand_x_axis.value
    cands_fig.yaxis.axis_label = cand_y_axis.value
    cands_fig.title.text = "%d candidates present" % len(df)
    source.data = dict(
        x=df[x_name],
        y=df[y_name],
        DM=df["DM"],
        snr=df["snr"],
        filter_width=df["logwidth"],
        sample=df["sample"],
        beam=df["beam"],
        # set color based on width:
        color=[filter_palette[width] for width in df["logwidth"]],
        # set alpha: 0.33 for S/N of 6, 1.0 for 10+
        alpha=[(snr-4.)/6. if snr <=10. else 1.0 for snr in df["snr"]],
    )

def tap_callback(attr, old, new):
    if len(new['1d']['indices']) > 0:
        cand_id = new['1d']['indices'][0]
        _cands = select_cands()
        selected_cand = _cands.iloc[cand_id]
        dm = selected_cand["DM"]
        sample = selected_cand["sample"]
        filter_ind = selected_cand["logwidth"]
        beam = selected_cand["beam"]
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
            #raise RuntimeError("Filterbank out-of-bound", "End window is out of bounds")
            print "Filterbank out-of-bound.", "End window is out of bounds", backstep, event_end
            backstep = int((fil.header.nsamples - sample)/tsamp_ms)
            event_end = int(backstep*2 + width)
            print "Adjusted backstep to", backstep, event_end

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

cand_controls = [cand_x_axis, cand_y_axis, cand_min_snr, cand_min_width,
        cand_max_width, cand_min_DM, cand_max_DM]
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
