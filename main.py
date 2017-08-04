import ConfigParser
import os, glob

import pandas as pd

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Div
from bokeh.io import curdoc
from bokeh.models.widgets import Select, Slider
from bokeh.layouts import layout, widgetbox
#choose palette for filter to color mapping. If more than 8 filter widths allowed, different palette will be needed
from bokeh.palettes import Spectral10

filter_palette = Spectral10

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

obs_moniker = config.get('cand_structure', 'observation_moniker')
obs_pattern = config.get('cand_structure', 'observation_pattern')

candidate_prefix = config.get('cand_structure', 'candidate_prefix')
candidate_postfix = config.get('cand_structure', 'candidate_postfix')
obs_pattern = config.get('cand_structure', 'observation_pattern')

obs_ids = []
cand_files = glob.glob(CAND_TOP_DIR + "/" + obs_pattern + "/" + candidate_prefix + '*' + candidate_postfix)
print "Found",len(cand_files),"candidates"
print "tried",CAND_TOP_DIR + "/" + obs_pattern + "/" + candidate_prefix + '*' + candidate_postfix
for cand_file_p in cand_files:
  cand_info = cand_file_p.split("/")
  obs_ids.append(cand_info[4])


print TOP_DIR+"/"+obs_pattern
#obs_ids = [os.path.basename(obs) for obs in glob.glob(TOP_DIR+"/"+obs_pattern)]
obs_selector = Select(title=obs_moniker, options=[obs_moniker] + sorted(obs_ids, reverse=True))

subobs_selector = Select()
subobservations_present = config.getboolean('cand_structure', 'subobservations')
subobs_moniker = config.get('cand_structure', 'subobservation_moniker')
subobs_pattern = config.get('cand_structure', 'subobservation_pattern')
if subobservations_present:
  subobs = [os.path.basename(subobs) for subobs in glob.glob(TOP_DIR + \
      "/" + obs_selector.value+"/" + subobs_pattern)]
  subobs_selector.title = subobs_moniker
  subobs_selector.options = sorted(subobs)

subsubobs_selector = Select()
subsubobservations_present = config.getboolean('cand_structure', 'subsubobservations')
subsubobs_moniker = config.get('cand_structure', 'subsubobservation_moniker')
subsubobs_pattern = config.get('cand_structure', 'subsubobservation_pattern')

if subsubobservations_present:
  subsubobs =[os.path.basename(subsubobs) for subsubobs in glob.glob(TOP_DIR +\
    obs_selector.value+"/" + subobs_selector.value + "/" +subsubobs_pattern)]
  subsubobs_selector.title = subsubobs_moniker
  subsubobs_selector.options = sorted(subsubobs)

def update_obs():
  if subobservations_present:
    subobs = [os.path.basename(subob) for subob in glob.glob(TOP_DIR+obs_selector.value+"/"+subobs_pattern)]
    subobs_selector.options=sorted(subobs)
    if subsubobservations_present:
      subsubobs = [os.path.basename(subsubob) for subsubob in
        glob.glob(TOP_DIR+obs_selector.value+"/" + subobs_selector.value + "/" + subsubobs_pattern)]
      subsubobs_selector.options = sorted(subsubobs)
  else:
    update_cand_file()

def update_subobs():
  antennas = [os.path.basename(cf) for cf in glob.glob(TOP_DIR+obs_selector.value+"/" + subobs_selector.value + "/" + subsubobs_pattern)]
  subsubobs_selector.options = sorted(antennas)

cands = pd.DataFrame()

candidate_prefix = config.get('cand_structure', 'candidate_prefix')
candidate_postfix = config.get('cand_structure', 'candidate_postfix')
def update_cand_file():
    cand_file = CAND_TOP_DIR + obs_selector.value + "/" # + subobs_selector.value + "." + subsubobs_selector.value + candidate_postfix
  if subobservations_present:
    cand_file += subobs_selector.value + "/"
    if subsubobservations_present:
      cand_file += subsubobs_selector.value + "/"
  cand_file += candidate_prefix + "*" + candidate_postfix
  #print cand_file
  cand_files = glob.glob(cand_file)#[0]
  if len(cand_files)  < 1 :
    print "Candidate file doesn't exit"
    print "tried:", cand_file
    return -1
  cand_file = cand_files[0]
  print cand_file
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
  "Beam No.": "beam",
  "Max S/N": "max_snr",
  "Primary Beam No.": "primary_beam",
  "No. of beams": "nbeams",
    "Sample No.": "sample",
    "Time (s)": "time",
    "log2(Boxcar width)": "logwidth",
    "DM trial": "dm_trial",
    "DM": "DM",
    "Member count": "members",
    "Begin (?)": "begin",
    "End (?)": "end",
    "Antenna": "antenna",
}

inverse_axis_map = {v: k for k, v in axis_map.iteritems()}

cand_x_axis = Select(title="X Axis", options=sorted(axis_map.keys()),
        value="Beam No.")
cand_y_axis = Select(title="Y Axis", options=sorted(axis_map.keys()),
        value="Time (s)")

cand_min_snr = Slider(title="Min S/N", value = 7.0, start=6.0, end=15.0, step=0.5)
cand_min_width = Slider(title="Min log2(width)", value = 0, start=0, end=10, step=1)
cand_max_width = Slider(title="Max log2(width)", value = 9, start=0, end=10, step=1)
cand_min_DM = Slider(title="Min DM", value = 20., start=0., end=4116., step=1)
cand_max_DM = Slider(title="Max DM", value = 4116., start=0., end=4116., step=1)
cand_min_beam = Slider(title="Min beam no.", value = 0, start=0., end=352, step=1)
cand_max_beam = Slider(title="Max beam no.", value = 352, start=0., end=352, step=1)

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data=dict(x=[], y=[], DM=[], snr=[], filter_width=[],
    sample=[], beam=[], color=[], alpha=[]))

source_ts = ColumnDataSource(data=dict(time=[], series=[]))
source_fb = ColumnDataSource(data=dict(image=[]))
source_fb_conv = ColumnDataSource(data=dict(image=[]))

source_for_table = ColumnDataSource(data=dict(time=[], snr=[], max_snr=[], beam=[],
  primary_beam=[], DM=[]))

columns = [
    TableColumn(field="time", title=inverse_axis_map["time"]),
    TableColumn(field="snr", title = inverse_axis_map["snr"]),
    TableColumn(field="max_snr", title = inverse_axis_map["max_snr"]),
    TableColumn(field="beam", title = inverse_axis_map["beam"]),
    TableColumn(field="primary_beam", title = inverse_axis_map["primary_beam"]),
    TableColumn(field="DM", title = inverse_axis_map["DM"])
]
candidate_table = DataTable(source=source_for_table, columns=columns, width=800)
table = widgetbox(candidate_table)

TOOLS = 'crosshair, box_zoom, reset, box_select, tap, hover'

cands_fig = figure(plot_height=600, plot_width=700, title="", tools = TOOLS,
        toolbar_location='right')
cands_plot = cands_fig.circle(x="x", y="y", source=source, size=15,
        color="color", line_color=None, fill_alpha="alpha")
cands_fig.text(x="x", y="y", text="beam", source=source, text_font_size='8pt',
        x_offset=-5, y_offset=5)

hover = cands_fig.select(dict(type=HoverTool))
hover.tooltips = [("S/N", "@snr"), ("DM", "@DM"), ("time", "@time"), ("sample", "@sample")]

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
            (cands.DM <= cand_max_DM.value) & 
            (cands.beam >= cand_min_beam.value) &
            (cands.beam <= cand_max_beam.value) 
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
        max_snr=df["max_snr"],
        filter_width=df["logwidth"],
        sample=df["sample"],
        beam=df["beam"],
        time=df["time"],
        # set color based on width:
        color=[filter_palette[int(width)] for width in df["logwidth"]],
        # set alpha: 0.33 for S/N of 6, 1.0 for 10+
        alpha=[(snr-4.)/6. if snr <=10. else 1.0 for snr in df["snr"]],
    )

def tap_callback(attr, old, new):
    if len(new['1d']['indices']) > 0:
        cand_id = new['1d']['indices'][0]
        _cands = select_cands()
        selected_cand = _cands.iloc[[cand_id]]
        dm = selected_cand["DM"].tolist()[0]
        sample = selected_cand["sample"].tolist()[0]
        time = selected_cand["time"].tolist()[0]
        filter_ind = selected_cand["logwidth"].tolist()[0]
        beam = selected_cand["beam"].tolist()[0]
        snr = selected_cand["snr"].tolist()[0]
        max_snr = selected_cand["max_snr"].tolist()[0]
        _, _dedisp_block, _conv_block, _time, _series = get_fbank_data_time(dm,
                time, 2**filter_ind, beam)
        source_ts.data["time"] = _time
        source_ts.data["series"] = _series
        source_fb.data["image"] = [_dedisp_block]
        source_fb_conv.data["image"] = [_conv_block]

    print "getting primary beam"
    primary_beam = selected_cand["primary_beam"].tolist()[0]
    max_snr = selected_cand["max_snr"].tolist()[0]

    print "Selected candidate:"
    print selected_cand

    print "new:"
    print new

    primary, rest = get_primary_and_rest_candidate(time, primary_beam, max_snr)
    if snr != primary["max_snr"].tolist()[0]:
      selected_cand = selected_cand.append(primary)
    selected_cand = selected_cand.append(rest)
    print type(selected_cand)
    print "Selected candidate after app:"
    print type(selected_cand)
    print selected_cand
    source_for_table.data = dict(
      time = selected_cand["time"],
      snr = selected_cand["snr"],
      max_snr = selected_cand["max_snr"],
      beam = selected_cand["beam"],
      primary_beam = selected_cand["primary_beam"],
      DM = selected_cand["DM"],
      logwidth = selected_cand["logwidth"]
    )

def tap_callback_table(attr, old, new):
  # like tap_callback but don't update the table
  if len(new['1d']['indices']) > 0:
    cand_id = new['1d']['indices'][0]
    print "tap_callback_table: cand_id", cand_id
    dm = source_for_table.data["DM"][cand_id]
    time = source_for_table.data["time"][cand_id]
    filter_ind = source_for_table.data["logwidth"][cand_id]
    beam = source_for_table.data["beam"][cand_id]
    _, _dedisp_block, _conv_block, _time, _series = get_fbank_data_time(dm,
        time, 2**filter_ind, beam)
    source_ts.data["time"] = _time
    source_ts.data["series"] = _series
    source_fb.data["image"] = [_dedisp_block]
    source_fb_conv.data["image"] = [_conv_block]


def get_primary_and_rest_candidate(time, primary_beam, max_snr):
  primary = cands[
    (cands.snr == max_snr) &
    (cands.beam == primary_beam) &
    (cands.primary_beam == primary_beam) &
    (np.abs(cands.time - time) <0.1) # TODO time separation as a parameter
  ]

  rest = cands[
    (cands.max_snr == max_snr) &
    (cands.snr < max_snr) &
    (cands.primary_beam == primary_beam) &
    (np.abs(cands.time - time) < 0.1)
  ]

  #source_for_table.data = dict(
    #time = primary["time"],
    #snr = primary["snr"],
    #max_snr = primary["max_snr"],
    #beam = primary["beam"],
    #primary_beam = primary["primary_beam"],
    #DM = primary["DM"]
  #)

  return primary, rest

filterbank_prefix = config.get('cand_structure', 'filterbank_prefix')

def get_fbank_data(dm, sample, width, beam):
    # based on Wael's filplot
    fil_pattern = (TOP_DIR+"/" + obs_selector.value+"/" + subobs_selector.value
      + "/" + subsubobs_selector.value + filterbank_prefix + "%03d/" % beam +"2*.fil")
    fil_fn = glob.glob(fil_pattern)
    if len(fil_fn) > 0:
      print fil_fn
      fil = FilReader(fil_fn[0])

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
    else:
      print "No filterbank found"
      print fil_pattern

def get_fbank_data_time(dm, _time, width, beam):
  print "get_fbank_data_time: running"
  # based on Wael's filplot
  fil_pattern = (TOP_DIR+"/" + obs_selector.value+"/" + subobs_selector.value
      + "/" + subsubobs_selector.value + filterbank_prefix + "%03d/" % beam +"2*.fil")
  fil_fn = glob.glob(fil_pattern)
  if len(fil_fn) > 0:
    print fil_fn[0]
    fil = FilReader(fil_fn[0])

    tsamp = fil.header.tsamp
    tsamp_ms = fil.header.tsamp*1000.

    sample = int(_time / tsamp)
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
  else:
    print "No filterbank found"
    print fil_pattern

cands_plot.data_source.on_change('selected', tap_callback)
candidate_table.source.on_change('selected', tap_callback_table)

top_level_controls = [ obs_selector]
if subobservations_present:
  top_level_controls.append(subobs_selector)
  if subsubobservations_present:
    top_level_control.append(subsubobs_selector)
obs_selector.on_change('value', lambda attr, old, new: update_obs())
if subobservations_present:
  subobs_selector.on_change('value', lambda attr, old, new: update_subobs())
  if subsubobservations_present:
    subsubobs_selector.on_change('value', lambda attr, old, new: update_cand_file())

cand_controls = [cand_x_axis, cand_y_axis, cand_min_snr, cand_min_width,
        cand_max_width, cand_min_DM, cand_max_DM, cand_min_beam, cand_max_beam]
for control in cand_controls:
    control.on_change('value', lambda attr, old, new: update())

sizing_mode = 'fixed'
top_level_inputs = widgetbox(*top_level_controls, sizing_mode=sizing_mode)
cand_control_inputs = widgetbox(*cand_controls, sizing_mode=sizing_mode)

desc = Div()
l = layout([[desc], [top_level_inputs], [cands_fig, cand_control_inputs],
    [timeseries_fig], [table], [dedisp_fig, conv_fig]])
curdoc().add_root(l)
curdoc().title = "Candidates"

url_args = curdoc().session_context.request.arguments
if "utc" in url_args.keys():
  obs_selector.value=url_args["utc"][0]
  update_cand_file()
