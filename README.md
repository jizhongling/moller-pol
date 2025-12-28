# moller-pol
Moller polarimeter DAQ analysis

* ROOT-based waveform analysis routine `AnaWaveform.C` to generate training ROOT files.
* Visualization and labeling tool `DrawWaveform.C` for spectra, summed channels, and PDF exports.
* Python training entrypoint `main.py` to load ROOT datasets, normalize features, and train clustering model over chunked file sets.
