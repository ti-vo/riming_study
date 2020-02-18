import toml
import numpy as np
import datetime
import sys
sys.path.append('../')
import functions
import functions.radar_functions as rf

def plot_spectra(case_study):

    # connect to larda
    larda = functions.pyLARDA.LARDA().connect(case_study['larda'], build_lists=True)
    # Read in the spectra
    dt_interval = [datetime.datetime.strptime(t, '%Y%m%d-%H%M') for t in case_study['time_interval']]
    rg_interval = case_study['range_interval']
    spectra = larda.read(case_study['radar'], case_study['spec_var'], dt_interval, rg_interval)

    # compute width
    widths = rf.denoise_and_compute_width(spectra)

    mask = np.full(spectra['var'].shape[0:2], False)
    criteria = case_study['conditions']
    if 'edge_width' in criteria:
        mask[np.logical_and(widths >= criteria['edge_width'][0], widths <= criteria['edge_width'][1])] = True

    numplots = 0
    while numplots < case_study['num_spec']:
        # pick random spectra fulfilling criterion
        r_y_i = [False, False]
        if not np.nansum(mask) == 0:
            while sum(r_y_i) == 0:
                r_x = np.random.choice(mask.shape[0])
                r_y_i = mask[r_x, :] == True
            r_y = np.random.choice(np.where(r_y_i)[0])
            random_spectrum = functions.pyLARDA.Transformations.slice_container(spectra, index={'time': [r_x],
                                                                                                'range': [r_y]})
            # plot them
            fig, ax = functions.pyLARDA.Transformations.plot_spectra(random_spectrum, z_converter='lin2z')
            fig.savefig(case_study['plot_dir']+f'spectrum_'
            f'{functions.h.ts_to_dt(random_spectrum["ts"]).strftime("%Y%m%d_%H%M")}_'
            f'{int(round(random_spectrum["rg"]))}_{widths[r_x, r_y]}.png')
            numplots += 1
            print(f'number of plots: {numplots}')
        else:
            print('no spectra fulfill criterion.')
            numplots = case_study['num_spec']


if __name__ == '__main__':

    config_case_studies = toml.load('../casestudies_toml/example_spectra_cases.toml')

    for case in config_case_studies['case']:
        case_study = config_case_studies['case'][case]
        print('case: ' + case)


        plot_spectra(case_study)

    print('\n ...Done...\n')
