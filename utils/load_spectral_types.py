from os.path import join
import vaex
import pathlib

current = pathlib.Path(__file__).parent.resolve()
root_data_dir = join(current, "..", "Data")


def load_spectral_types():
    file_spectral_class = join(root_data_dir, "mamajek-spectral-class.hdf5")
    df_spectral_class = vaex.open(file_spectral_class)
    df_filtered = df_spectral_class[["SpT", "M_J", "J-H", "H-Ks"]]
    mask = (df_filtered["M_J"] == df_filtered["M_J"])*(df_filtered["H-Ks"]
                                                       == df_filtered["H-Ks"])*(df_filtered["J-H"] == df_filtered["J-H"])
    df_filtered_good = df_filtered[mask].to_pandas_df()
    df_filtered_good['J-K'] = df_filtered_good['J-H']+df_filtered_good['H-Ks']
    return df_filtered_good
