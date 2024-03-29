# LISAsky: 2D and 3D Sky Maps Generation for LISA data
This project contains code for generating 3D sky maps using FITS (Flexible Image Transport System) files. The codes currently are based on ```ligo.skymap``` package, but they will further extended for LISA objectives. This code is part of my work as LISA-submissions data analyst of [Sangria](https://lisa-ldc.lal.in2p3.fr/challenge2a) SMBHB training set.
I would like to thank Dr. Quentin Baghi for his comments and suggestions.

## File Description

- `3Dmaps.ipynb` : This is a Jupyter notebook that contains an example for generating the 3D sky maps.
- `sky_utils.py` : Utilities for the sky maps (you can choose between 'ICRS' and 'Galactic' coordinates - by default is ```frame = 'icrs'```)

## How to Run

1. Follow the instructions to create the conda environment. 
2. Open the `3Dmaps.ipynb` file in a Jupyter notebook environment.
3. Run the `plot_skymaps` function with the appropriate parameters (modify the directory paths in your needs).

## Function Parameters

- `ra`, `dec`, `dist`: These are the coordinates for the 3D sky map.
- `chains`: This is an optional parameter in case that you want to plot also some samples.
- `dirname`: This is the directory name where the FITS file will be located (or already lives in).
- `name`: This is the ID of the sky map.
- `fits_filename`: This is the filename of the FITS file.
- `figwidth`: This is the width of the figure to be plotted.
- `dpi`: This is the resolution of the figure.
- `contour_levels`: This is an optional parameter for the contour levels of the plot.
- `transparent`: This is a boolean parameter to make the plot transparent.

## Dependencies
First, let's make our conda environment and install the needed packages to run this notebook:

```conda create -n LISA_LDC python=3.8```

```conda activate LISA_LDC```

```pip install lisa-data-challenge``` 
(In case of installation failed, you may have to ```conda install gsl``` and/or ```conda install fftw```)

```pip install ChainConsumer```

```python -m pip install corner```

```conda install ipykernel```

```pip install lisacattools```

```ipython kernel install --user --name=LDC```

