import numpy as np
import sys
from os.path import join, abspath

# Initial position
ra0 = 0
ra1 = 30
start_dec = -999

ras = np.arange(ra0,ra1+0.1, 1).astype(int)
decs = np.arange(-90,90+0.1,1).astype(int)

# =================================================
# =================================================
# =================================================
# import utils
top_dir = join(abspath("../../"))
if not top_dir in sys.path:
    sys.path.insert(0, top_dir)

from utils import append_name, iterate_job, safe_mkdir

# =================================================
# paths
# get the root of data directory
root_data_dir = abspath(join(top_dir, "Data"))

# Create a directory for Gaia DR3 and 2MASS data
# or if it already exists, just move on
name = "Gaia-2MASS-5"
gaia_data_dir = join(root_data_dir, name)
safe_mkdir(gaia_data_dir)

# Do the same for 2MASS data
name = "TWOMASS-5"
tmass_data_dir = join(root_data_dir, name)
safe_mkdir(tmass_data_dir)

# =================================================
# Initialize the columns and tables
# Gaia
# =================================================
column_gaia = [
    "source_id", 
    "pm", "pmra", "pmra_error AS e_pmra", "pmdec", "pmdec_error AS e_pmdec", 
    "parallax", "parallax_error AS e_parallax", 
    "phot_g_mean_mag AS Gmag",	"phot_bp_mean_mag AS BPmag", "phot_rp_mean_mag AS RPmag", 
    "phot_bp_mean_flux_over_error AS Fb_over_err", 
    "phot_rp_mean_flux_over_error AS Fr_over_err", 
    "ruwe", 
    "phot_bp_rp_excess_factor AS excess_factor", 
    "radial_velocity AS rv_gaia", "radial_velocity_error AS e_rv_gaia",
    "l AS glon", "b AS glat", 
    "teff_gspphot", "teff_gspphot_lower", "teff_gspphot_upper",
    "logg_gspphot", "logg_gspphot_lower", "logg_gspphot_upper"
]

column_astrophysical = [
    "mh_gspphot", "mh_gspphot_lower", "mh_gspphot_upper", 
    "distance_gspphot", "distance_gspphot_lower", "distance_gspphot_upper", 
    "ag_gspphot", "ag_gspphot_lower", "ag_gspphot_upper",
    "mh_gspspec", "mh_gspspec_lower", "mh_gspspec_upper", 
    "alphafe_gspspec", "alphafe_gspspec_lower", "alphafe_gspspec_upper", 
    "fem_gspspec", "fem_gspspec_lower", "fem_gspspec_upper" , 
    "spectraltype_esphs"
]

column_join = ["original_psc_source_id AS tmass"]

gaia_alias = "gdr3"
astrophysical_alias = "astrophysical"
join_alias = "join_table"

column_gaia = list(map(lambda x: append_name(x, gaia_alias), column_gaia))
column_astrophysical = list(map(lambda x: append_name(x, astrophysical_alias), column_astrophysical))
column_join = list(map(lambda x: append_name(x, join_alias), column_join))

columns = column_gaia + column_astrophysical + column_join

# =================================================
#  2mass
# =================================================
columns_tmass = ["ra", "dec","j_m", "k_m", "designation", "ph_qual"]

# rename the table columns as 
columns_tmass_names = ["ra", "dec","jmag", "kmag", "designation", "ph_qual"]

tmass_table = "fp_psc"
column_tmass = list(map(lambda x: append_name(x, tmass_table), columns_tmass))

# =================================================
# The query
# =================================================

TOP = 100_000

gaia_query = f"""
SELECT TOP {TOP} {', '.join(columns)}
FROM gaiadr3.gaia_source AS {gaia_alias}
LEFT JOIN gaiadr3.astrophysical_parameters AS {astrophysical_alias} ON astrophysical.source_id = gdr3.source_id
RIGHT JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS tmass ON tmass.source_id = gdr3.source_id
RIGHT JOIN gaiadr3.tmass_psc_xsc_join as {join_alias} ON join_table.clean_tmass_psc_xsc_oid = tmass.clean_tmass_psc_xsc_oid
"""

tmass_query = f"""
SELECT TOP {TOP} {", ".join(columns_tmass)} 
FROM {tmass_table}
"""

gaia_log = gaia_data_dir
tmass_log = tmass_data_dir

iterate_job(ras, decs, gaia_query, tmass_query, gaia_log, tmass_log, columns_tmass_names, TOP=TOP, write=True, start_dec=start_dec)