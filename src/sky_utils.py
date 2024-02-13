################ *** General for the plots *** ################
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
%matplotlib inline

# plt.style.use(['seaborn-ticks','seaborn-deep'])
plt.style.use(['seaborn-v0_8-ticks','seaborn-v0_8-deep'])

rcparams = {
          'axes.labelsize': 8,
          'font.size': 10,
          'legend.fontsize': 8,
          'xtick.color': 'k',
          'xtick.labelsize': 8,
          'ytick.color': 'k',
          'ytick.labelsize': 8,
          'text.usetex': False,
          'font.family': 'Times New Roman',
          'font.sans-serif': 'Bitstream Vera Sans',
          'mathtext.fontset': 'stixsans',
          'text.color': 'k',
          'figure.figsize': [12, 7],
          'figure.dpi': 900,
          'axes.grid' : True
          }

rcparams['axes.linewidth'] = 0.5

mpl.rcParams.update(rcparams)
############################################################

def write_maps_fits(pts, dirname, name, trials=5, jobs=8):
    """ Generate a fits file and sky map from posteriors

        Code adapted from ligo.skymap.
        Installation of ligo.skymap is required. 

        Parameters
        ----------
        pts: np.array
            Posterios of sky location (in rad) and distance.
            For example, pts = np.column_stack((post_ra, post_dec, distances))
        dirname: str
            Name of directory to save the files
        name: str
            Under which name/tag to save the file
        trials: int
            Number of trials at each clustering number
        jobs: int
            Number of multiple threads
        """
    try:
        from astropy.time import Time
        from ligo.skymap.kde import Clustered2Plus1DSkyKDE
        import pickle
        from ligo.skymap import io
        import healpy as hp
        import os
        import numpy as np
    except ImportError as e:
        logger.info("Unable to generate skymap: error {}".format(e))
        return
    # Create the directory if it doesn't exist
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # save skypost in pkl under the name of default_obj_filename
    default_obj_filename = dirname+'/testing_3D_{}.pkl'.format((str(name)))
    if not os.path.exists(default_obj_filename):
        # clustering
        cls = Clustered2Plus1DSkyKDE
        skypost = cls(pts, trials=trials, jobs=jobs)
        with open(default_obj_filename, 'wb') as out:
            pickle.dump(skypost, out)
    else:
        print('Loading existing skypost')
        with open(default_obj_filename, 'rb') as inp:
            skypost = pickle.load(inp)
    # generate healpix skymap
    hpmap = skypost.as_healpix()
    # write meta
    #hpmap.meta.update(io.fits.metadata_for_version_module(version))
    hpmap.meta['creator'] = "A.Sasli"
    hpmap.meta['origin'] = 'LDC LISA'
    hpmap.meta['gps_creation_time'] = Time.now().gps
    hpmap.meta['history'] = ""
    hpmap.meta['distmean'] = np.mean(pts[:,-1])
    hpmap.meta['diststd'] = np.std(pts[:,-1])
    # write sky map in fits format
    fits_filename = os.path.join(dirname,"{}_skymap.fits".format(name))
    io.write_sky_map(fits_filename, hpmap, nest=True)

def plot_skymaps(ra, dec, dist, chains, 
                 dirname, name, fits_filename,
                 figwidth=3.5, dpi=300, 
                 contour_levels=[50, 90], transparent=True):
    """Generate 3D and 2D sky map from fits file.

    Parameters
    ----------
    ra : float
        True right ascension in rad.
    dec : float
        True declination in rad.
    dist : float
        True distance.
    chains : dictionary, optional
        Posteriors of ra, dec (in rad) and distance.
        If given, then chains will be plotted.
    dirname : str
        Name of directory to save the files.
    name : str
        Under which name/tag to save the file.
    fits_filename : str
        Name of fits file.
    dpi : int, optional
        Resolution of figure in dots per inch. Default is 300.
    figwidth : int, optional
        Width of figure. Default is 3.5.
    contour_levels : list, optional
        List of contour levels to use. Default is [50, 90].
    transparent : bool, optional
        Save image with transparent background. Default is True.
    """
    try:
        # Import required libraries
        from ligo.skymap import io, plot, postprocess
        import healpy as hp
        from ligo.skymap.core import volume_render, marginal_pdf
        from ligo.skymap.distance import principal_axes, parameters_to_marginal_moments
        from ligo.skymap.plot import marker
        from matplotlib import gridspec
        import astropy_healpix as ah
        from astropy import units
        import numpy as np
        import seaborn
        import os
        import matplotlib.pyplot as plt
    except ImportError as e:
        logger.info("Unable to generate skymap: error {}".format(e))
        return

    # Read sky map
    skymap, metadata = io.fits.read_sky_map(fits_filename, nest=None)
    (prob, mu, sigma, norm), metadata = io.read_sky_map(fits_filename, distances=True)

    # Start generating 3D map
    npix = len(prob)
    nside = ah.npix_to_nside(npix)
    mean, std = parameters_to_marginal_moments(prob, mu, sigma)
    max_distance = mean + 2.5 * std
    R = np.ascontiguousarray(principal_axes(prob, mu, sigma))

    # Whether to plot different chains
    if chains is not None:
        chain = np.dot(R.T, (hp.ang2vec(
             0.5 * np.pi - chains['dec'], chains['ra']) *
             np.atleast_2d(chains['distances']).T).T)
        
    print('Starting volumetric image...')

    # Color palette for markers
    colors = seaborn.color_palette(n_colors=3 + 1)
    truth_marker = marker.reticle(
        inner=0.5 * np.sqrt(2), outer=1.5 * np.sqrt(2), angle=45)

    fig = plt.figure(frameon=False)
    n = 2
    gs = gridspec.GridSpec(
        n, n, left=0.01, right=0.99, bottom=0.01, top=0.99,
        wspace=0.05, hspace=0.05)

    imgwidth = int(dpi * figwidth / n)
    s = np.linspace(-max_distance, max_distance, imgwidth)
    xx, yy = np.meshgrid(s, s)

    print('> Marginalize onto the given faces...')
    for iface, (axis0, axis1, (sp0, sp1)) in enumerate((
            (1, 0, [0, 0]),
            (0, 2, [1, 1]),
            (1, 2, [1, 0]),)):
    
        print('>> Plotting projection {0}'.format(iface + 1))
        density = volume_render(
            xx.ravel(), yy.ravel(), max_distance, axis0, axis1, R, False,
            prob, mu, sigma, norm).reshape(xx.shape)
    
        # Plot heat map
        ax = fig.add_subplot(gs[sp0, sp1], aspect=1)
        ax.imshow(
            density, origin='lower',
            extent=[-max_distance, max_distance, -max_distance, max_distance],
            cmap="cylon")
    
        # Add contours
        if contour_levels is not None:
            flattened_density = density.ravel()
            indices = np.argsort(flattened_density)[::-1]
            cumsum = np.empty_like(flattened_density)
            cs = np.cumsum(flattened_density[indices])
            cumsum[indices] = cs / cs[-1] * 100
            cumsum = np.reshape(cumsum, density.shape)
            u, v = np.meshgrid(s, s)
            contourset = ax.contour(
                u, v, cumsum, levels=contour_levels, linewidths=0.5)

        # Mark locations
        ax._get_lines.get_next_color()  # skip default color
        theta = 0.5 * np.pi - dec
        phi = ra
        xyz = np.dot(R.T, (hp.ang2vec(
             0.5 * np.pi - dec, ra) *
             np.atleast_2d(dist).T).T)
        ax.plot(xyz[axis0], xyz[axis1], marker=truth_marker,
                markerfacecolor='none', markeredgewidth=1)
    
        # Plot chain
        if chains is not None:
            ax.plot(chain[axis0], chain[axis1], '.k', markersize=0.1)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Set axis limits
        ax.set_xlim([-max_distance, max_distance])
        ax.set_ylim([-max_distance, max_distance])

        # Mark origin (Earth)
        ax.plot([0], [0], marker=marker.earth, markersize=5,
            markerfacecolor='none', markeredgecolor='black',
            markeredgewidth=0.75)
    
        if iface == 2:
            ax.invert_xaxis()
        
            # Add contour labels if contours requested
            if contour_levels is not None:
                ax.clabel(contourset, fmt='%d%%', fontsize=7)

            ax.plot([0.0625, 0.3125], [0.0625, 0.0625],
                 color='black', linewidth=1, transform=ax.transAxes)
            ax.text(0.0625, 0.0625,
                 '{0:d} Mpc'.format(int(np.round(0.5 * max_distance))),
                 fontsize=8, transform=ax.transAxes, verticalalignment='bottom')

    # Create marginal distance plot.
    print('Plotting distance')
    gs1 = gridspec.GridSpecFromSubplotSpec(5, 5, gs[0, 1])
    ax = fig.add_subplot(gs1[1:-1, 1:-1])

    # Plot marginal distance distribution, integrated over the whole sky.
    d = np.linspace(0, max_distance)
    ax.fill_between(d, marginal_pdf(d, prob, mu, sigma, norm),
                alpha=0.5, color=ax._get_lines.get_next_color())

    # Scale axes
    ax.set_xticks([0, max_distance])
    ax.set_xticklabels(['0', "{0:d}\nMpc".format(int(np.round(max_distance)))],
            fontsize=9)
    figs_dir = dirname + '/figs/'
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    figname = figs_dir+'volume_skymap_'+str(name)+'.pdf'
    plt.savefig(figname, transparent=transparent)
    print('Image saved at {}'.format((str(figname))))
    
    print('Plotting 2D map in astro hours mollweide projection')
    fig = plt.figure(frameon=False)
    nside = hp.npix2nside(len(skymap))
    # Convert sky map from probability to probability per square degree.
    deg2perpix = hp.nside2pixarea(nside, degrees=True)
    probperdeg2 = skymap / deg2perpix
    ax = plt.axes(projection='astro hours mollweide')
    ax.grid()

    # Plot sky map.
    vmax = probperdeg2.max()
    img = ax.imshow_hpx((probperdeg2, 'ICRS'),
                            nested=metadata['nest'],
                            vmin=0.,
                            vmax=vmax,
                            cmap="cylon")
    cb = plot.colorbar(img)
    cb.set_label(r'prob. per deg$^2$')
    confidence_levels = 100 * postprocess.find_greedy_credible_levels(skymap)
    if contour_levels is None:
        contour_levels = [50, 65]
    contours = ax.contour_hpx((confidence_levels, 'ICRS'),
                                      nested=metadata['nest'],
                                      colors='k',
                                      linewidths=0.5,
                                      levels=contour_levels)
    fmt = r'%g%%'
    plt.clabel(contours, fmt=fmt, fontsize=6, inline=True)
     # Add a white outline to all text to make it stand out from the background.
    plot.outline_text(ax)
    pp = np.round(contour_levels).astype(int)
    ii = np.round(np.searchsorted(np.sort(confidence_levels), contour_levels) *deg2perpix).astype(int)
    text = []
    for i, p in zip(ii, pp):
        text.append(u'{:d}% area: {:d} deg$^2$'.format(p, i))
    ax.text(1, 1, '\n'.join(text), transform=ax.transAxes, ha='right')

    ax.scatter((ra*units.rad).to_value(units.deg), 
               (dec*units.rad).to_value(units.deg), 
               color="darkorchid", marker='x', 
               transform=ax.get_transform('world'))

    figname = figs_dir + 'skymap_' + str(name) + '.pdf'
    plt.savefig(figname, transparent=transparent)
    print('Image saved at {}'.format((str(figname))))