# NOTES
Author: C. Walker

This repository contains a limited copy of the working directory used to generate and process data for the project: "The Dispersion Measure Contributions of the Cosmic Web" on the Raven MPCDF supercomputing cluster.

- Jupyter notebooks, python scripts, and shell scripts used to generate and process data are preserved in this backup.
- The resulting data files and plots generated by said scripts and notebooks are generally very large (of order tens (or more) of GB) or numerous, thus these are not included in this backup.
- However the original directory structure is preserved for easier recreation of said data if necessary
- Currently, `work_directory/Paper_Plots_Pipe.ipynb` is ignored (in `.gitignore`) as its ~500MB size exceeds github's ~100MB upload limits.

The original directory containing this work is at /u/cwalker/Illustris_Zhang_Method/ on raven.

---

# NECESSARY DEPENDENCIES
I.e. Things which will probably be necessary to run these codes/create new data.

- the YT virtual environment from https://github.com/mbcxqcw2/Python_Virtual_Environments
- a copy of IllustrisTNG (tng-project.org/). I used the one at /virgotng/universe/IllustrisTNG/ on raven.
- adding the `work_directory/charlie_TNG_lib` directory to your pythonpath.

---

# CODE WHICH GENERATES DATA FOR PAPER FIGS/TABLES

- The pipes/los segments themselves are created with:
  - `work_directory/batch_jobs/batch_scripts/make_pipes_5.py`
  - associated `.sh` scripts
  - Note: for speed, these are created sith temporary, placeholder subhaloIDs.
  - The resulting pipes are output to `work_directory/SpeedTest/` or `work_directory/SpeedTempTest/`.
- The true subhaloID lists associated with the above created pipes are created with:
  - `batch_jobs/batch_scripts/get_subhaloIDs_1.py`
  - associated `.sh` scripts
  - subhalo/particle ID matchlists created with `work_directory/Create_Particle_Subhalo_Matchlists_5.ipynb`
  - The resulting subhaloID lists are output to `work_directory/SpeedTempTest/`

- Figs.:
  - Fig. 1: `Paper_Plots_TNG_Slice_Structure.ipynb`
  - Fig. 2: `work_directory/Paper_Plots_Ne_Analysis.ipynb`
  - Fig. 3: `work_directory/Paper_Plots_Ne_Analysis.ipynb`
  - Fig. 4: `work_directory/Paper_Plots_Pipe_Structure_Subhalos.ipynb`
  - Fig. 5: `work_directory/Paper_Plots_Fitting_3.ipynb`
  - Fig. 6: `work_directory/Paper_Plots_DMs.ipynb`
  - Fig. 7: `work_directory/Pipe_LSS_Analysis_5.ipynb`
  - Fig. 8: `work_directory/Paper_Plots_IF_vs_DM_3.ipynb`
  - Fig. 9: `work_directory/Paper_Plots_Subhalo_Analysis_By_Mass.ipynb`
  - Fig. 10: `work_directory/Paper_Plots_Subhalo_Analysis_By_Mass.ipynb`

- Tables:
  - Table 3: `work_directory/Paper_Plots_Fitting_3.ipynb`
  - Table 4: `work_directory/Paper_Plots_DMs.ipynb`,`work_directory/Pipe_LSS_Analysis_5.ipynb`
  - Table 5: `work_directory/Paper_Plots_Subhalo_Analysis_By_Mass.ipynb`
  - Table 6: `work_directory/Paper_Plots_Subhalo_Analysis_By_Mass.ipynb`
  - Table 7: `work_directory/Paper_Plots_Subhalo_Analysis_By_Mass.ipynb`

