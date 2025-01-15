# this is a python script to get model_listing

import os

def main(model_list, cmip_v, mip_name, exp_name, var_name, frequency):
    i = 1
    for mod_name in model_list:
        # run clef command to get the listing
        os.system('clef --local ' + cmip_v + ' -mip ' + mip_name + ' -e ' + exp_name + ' -m ' + mod_name + ' -v ' + var_name + ' -t ' + frequency + ' --csv')
        # rename the csv file
        os.system('mv CMIP6_query.csv ' + mod_name + '_' + var_name + '.csv')
        print(f'finished {i}/{len(model_list)} : ' + mod_name)
        i += 1
    print('Completed all models')

if __name__ == '__main__':
    # model_list = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-CM-1-1-MR', 'BCC-CSM2-MR', 'BCC-ESM1', 'CESM2-WACCM', 'CESM2', 'CIESM', 'CMCC-CM2-SR5', 'CanESM5', 'E3SM-1-1', 'EC-Earth3-Veg-LR', 'EC-Earth3-Veg', 'EC-Earth3', 'FGOALS-f3-L', 'FGOALS-g3', 'FIO-ESM-2-0', 'GFDL-CM4', 'GFDL-ESM4', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'TaiESM1', 'UKESM1-0-LL']
    # model_list = ['IITM-ESM', 'CAMS-CSM1-0', 'NorCPM1', 'CAS-ESM2-0', 'SAM0-UNICON', 'GISS-E2-1-H', 'GISS-E2-1-G', 'MCM-UA-1-0', 'BCC-CSM2-MR', 'BCC-ESM1']
    model_list = ['AWI-ESM-1-1-LR', 'CESM2-FV2', 'CESM2-WACCM-FV2', 'CIESM', 'CMCC-CM2-HR4', 'E3SM-1-0', 'E3SM-1-1-ECA', 'EC-Earth3-Veg', 'FGOALS-f3-L', 'FGOALS-g3', \
                  'GISS-E2-1-G-CC', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 'MPI-ESM-1-2-HAM']
    cmip_v = 'cmip6'
    mip_name = 'CMIP'
    exp_name = 'historical'
    var_name = 'tos'
    frequency = 'Omon'
    main(model_list, cmip_v, mip_name, exp_name, var_name, frequency)
