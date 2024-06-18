from pathlib import Path
import pandas as pd



aircraft_path = Path(r'D:\Python\TropSF6\Data')

# missions = ["tacts", "wise", "southtrac"]
missions = ["tacts", "southtrac"]
instruments = ['ECD', 'MS']


for instr in instruments:

    Fdata = {}
    for m in missions:
        fname = f'{m.upper()}_{instr}.csv'
        data = pd.read_csv(Path(aircraft_path, fname), sep=';')
        data.columns = [x.lower() for x in data.columns]
        Fdata[m] = data

    df_merge = pd.concat([Fdata['tacts'], Fdata['southtrac']])

    df_merge.to_csv(rf'D:\Python\TropSF6\Data\ghost_merge_{instr}.csv', index=False, na_rep='NaN', sep=';')
