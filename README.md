## Simulation

To investigate the behavior of the detector system and visualize the effects of different experimental parameters, we developed a Python simulation (photon_gui.py). Note that AI was used in coding this simulation.

### Signal and laser

The simulation produces a histogram of photons per time bin. Signal photons are Poisson-distributed, with a mean that can be set by the pulse shape and the signal rate. The rate can be set manually or computed from the included Mie/Rayleigh scattering model, which can then be interpreted as either the peak rate or the average rate over the window. The simulation features four pulse shapes (logistic edges, square, logistic, and Gaussian), which can be used to model the laser pulse shape.

When the pixel view window is enabled, the histogram is converted to a single pixel timetrace, which shows the number of photons per bin as a brightness value on the pixel view. This way, you can see what each pixel might look like in the real setup and whether the change in brightness is noticeable compared to the noise level.

When you generate a histogram, the runs are produced in chunks. The number of runs per chunk grows geometrically to provide a clear picture of how the histogram develops over time. During this process, all events go into one time-ordered heap, so dead time and afterpulses can be processed in true chronological order. Besides, each detection event is also tagged as signal, dark, afterpulse, or noise, so the histogram can show each component separately.

The simulation also calculates analytic expectation curves for the probability per bin. It does so by convolving the pulse and background with the afterpulse delay using FFTs and then smearing them with jitter. The expected total curve and ±√N deviation band (which can be enabled in the display tab) are calculated using this method. These are then plotted against the histogram so that the simulation can be compared to the theory.

### Background noise and detector effects

The simulation includes three other settings that can affect the signal rate. These are background noise, dark counts, and afterpulsing, and can be set manually. The afterpusling setting determines the chance afterpulses are spawned when a detection event happens, and the average delay.

More importantly, the simulation also allows for modeling the photon deadtime loss, which will reject events that are closer to the last accepted event than the dead time. The deadtime can be manually set and also carries across run boundaries.

There is also a setting to enable timing jitter, which will shift detection events using a Gaussian distribution. This setting must be set according to the detector jitter FWHM.
