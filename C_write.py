import C_read
import C_tools
import numpy as np
from pathlib import Path



# checks for newest version number and returns filename with version number increased by 1
def next_version_number(path, flight, prefix, high_res=False, verbose=True):
    prefix = prefix.upper()
    fname_path = C_read.find_most_recent(path, flight, prefix, high_res=high_res, verbose=verbose)

    create_first = False
    if fname_path is None:
        fname_path = C_read.find_most_recent(path, flight, 'MA', high_res=True, verbose=verbose)
        if fname_path is None:
            return None
        else:
            create_first = True
            print('File name constructed from MA high resolution file.')

    fname = fname_path.split('\\')[-1]
    version_str = fname.split('.')[0]
    version_str = version_str.split('_')[-1]

    if not create_first:
        version = np.int(version_str.lower().replace('v', ''))

        version_out = version + 1
        version_str_out = 'V' + str(version_out).zfill(2)
    else:
        version_str_out = 'V01'
    fname_path_out = fname_path.replace(version_str, version_str_out)
    if create_first:
        fname_path_out = fname_path.replace('MA', prefix)

    if verbose:
        print('Next file:')
        print(fname_path_out)
    return fname_path_out


# %%
def write_traj_for_IGOR(TrajData):
    out_path = Path('D:/CARIBIC/Trajectories/554')
    file_name_start = "WAS_traj_F554"  # no extension
    out_file = (out_path/file_name_start)

    sample_no_str = list(TrajData.keys())
    sample_no_str.remove('headers')

    for i, x in enumerate(sample_no_str):
        tmp_dict = {}
        tmp = TrajData[x]
        for j, y in enumerate(tmp):
            tmp_dict[f't_day_{j}'] = [int(text)/1440 for text in y['TTTT'].tolist()]
            tmp_dict[f'lon_{j}'] = y['lon'].tolist()
            tmp_dict[f'lat_{j}'] = y['lat'].tolist()
            tmp_dict[f'p_{j}'] = y['p'].tolist()

        f = open(f"{out_file}_{x}.txt", 'w')
        f.write('\t'.join(list(tmp_dict.keys())))
        f.write('\n')

        for k in range(len(tmp_dict[next(iter(tmp_dict))])):
            for key in tmp_dict:
                f.write(f'{tmp_dict[key][k]:.2f}\t')
            f.write('\n')
        f.close()

    f = open("sample_locs.txt", 'w')
    f.write('lat_start\tlon_start\n')
    for i in range(len(TrajData['headers'])):
        lat = (TrajData['headers'][i]['lat']).tolist()[0]/10.
        lon = (TrajData['headers'][i]['lon']).tolist()[0]/10.

        f.write(f'{lat:.2f}\t{lon:.2f}\n')

    f.close

    return tmp_dict


#%%
def data_export(df_flights, Fdata, var_list, flight_numbers, df_return=False, write=False):
    if var_list is None:
        var_list = ['co2', 'ch4', 'n2o', 'sf6',
                    'hfc_125', 'hfc_134a', 'halon_1211', 'cfc_12', 'hcfc_22',
                    'd_co2', 'd_ch4', 'd_n2o', 'd_sf6',
                    'd_hfc_125', 'd_hfc_134a', 'd_halon_1211', 'd_cfc_12', 'd_hcfc_22',
                    'int_co',
                    'year_frac', 'lat', 'lon', 'p', 'int_tpot', 'int_o3']
    tmp_df = bla = C_tools.extract_data(df_flights, Fdata, var_list, flight_numbers)

    if write:
        tmp_df.to_csv(r'D:\Python\TropSF6\Data\CARIBIC.csv', index=False, na_rep='NaN', sep=';')
    if df_return:
        return tmp_df
