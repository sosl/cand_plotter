# FRB candidate viewer

This code can be used for plotting FRB candidates. Currently it supports candidates from Heimdall and can be easily modified to work with other search pipelines such as Fredda.

### Prerequisites

```
pandas
bokeh
sigpyproc
mbplotlib
```

### Deploying

Simply deploy on your system and create a ~/.candplotter.cfg file based on the included example and run a bokeh server.
```
bokeh serve main.py
```

## Contributing

Any contributions welcome.

## Authors

* **Stefan Oslowski** - *Initial work* - [homepage](https://astronomy.swin.edu.au/~soslowski)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* get_fbank_data is based on Wael's filplot

