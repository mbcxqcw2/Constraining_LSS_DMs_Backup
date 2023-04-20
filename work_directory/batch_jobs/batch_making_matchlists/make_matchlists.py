#Notes
#This is a script for use as a batch job
#It is built using /u/cwalker/Illustris_Zhang_Method/Create_Particle_Subhalo_Matchlists_2.ipynb
#It creates subhalo ID and particle ID lists for snapshots for IllustrisTNG simulations.
#These can be loaded and used for cross-matching for impact factors to galaxies.

#########
#IMPORTS#
#########

import os
import yt
import h5py
import random

import numpy as np
import illustris_python as il

from matplotlib import pyplot as plt
from charlie_TNG_tools import pID2shID


############
#INITIALISE#
############

#simulation to create lists for
sim_to_use = 'TNG300-1'

#snapshots to create lists for
snap_list = [91,84,78,72,67,59,50,40,33,25,21,17]

#base path to data
basePath = '/ptmp/cwalker/Illustris_FRB_Project/TNG_copies/virgo/simulations/IllustrisTNG/{0}/output/'.format(sim_to_use)

#path to simulation hdf5 file
simPath = '/ptmp/cwalker/Illustris_FRB_Project/TNG_copies/virgo/simulations/IllustrisTNG/{0}/simulation.hdf5'.format(sim_to_use)

#check these exist
print('Testing whether basePath and simPath exist...')
print('basePath exists = {0}'.format(os.path.exists(basePath)))
print('simPath exists = {0}'.format(os.path.exists(simPath)))

########################################
#CREATE DIRECTORIES TO STORE MATCHLISTS#
########################################

#check to see if a top directory exists
topdir_name = '/u/cwalker/Illustris_Zhang_Method/Sim_Matchlists/'
topdir_check = os.path.exists(topdir_name)

#if it exists, print
if topdir_check == True:
    print('Top directory ({0}) already exists.'.format(topdir_name))
    
#else, create it.
elif topdir_check == False:
    print('Creating top directory for matchlists at {0}...'.format(topdir_name))
    os.mkdir(topdir_name)
    print('{0} created.'.format(topdir_name))
    
    
#check to see if subdirectory for particular simulation matchlist exists
subdir_name = topdir_name+'Matchlist_dir_{0}'.format(sim_to_use)
subdir_check = os.path.exists(subdir_name)

#if it exists, print
if subdir_check == True:
    print('Directory to hold {0} matchlist ({1} exists.)'.format(sim_to_use,subdir_name))

#else, create it
elif subdir_check == False:
    print('Creating subdirectory {0}...'.format(subdir_name))
    os.mkdir(subdir_name)
    print('{0} created.'.format(subdir_name))
    
###########################
#CREATE DESIRED MATCHLISTS#
###########################

#loop over snapshots
for snapshot_number in snap_list:
    print('Creating for {0} snap {1}'.format(sim_to_use,snapshot_number))

     #########################
     #SNAPSHOT DATA LOCATIONS#
     #########################

    offsetFile = '/virgo/simulations/IllustrisTNG/{0}/postprocessing/offsets/offsets_0{1}.hdf5'.format(sim_to_use,snapshot_number)
    data_loc = '/virgo/simulations/IllustrisTNG/{0}/output/snapdir_0{1}/snap_0{1}.0.hdf5'.format(sim_to_use,snapshot_number)
    partIDs_loc = '/Snapshots/{0}/PartType0/ParticleIDs'.format(snapshot_number)
    
    print('Processing snapshot {0} at:\n{1}\n with offset file:\n{2}\n and particle IDs file loc:\n{3}'.format(snapshot_number,data_loc,offsetFile,partIDs_loc))

    #########################
    #CREATE PARTICLE ID LIST#
    #########################
    
    #get all gas particle IDs in snapshot
    with h5py.File(simPath) as f:
        print('simpath is {0}'.format(simPath))
        print('f is {0}'.format(f))
        allparts = f[partIDs_loc][:]
    
    #create a list version of every particle ID
    AllPartList = allparts.tolist()
    
    ########################
    #CREATE SUBHALO ID LIST#
    ########################
    
    #load and index cell data
    ds=yt.load(data_loc)
    ds.index
    
    #choose what to ID
    to_ID = np.arange(ds.particle_type_counts['PartType0'])
    partType = 0 #gas
    
    #choose subhalo fields to load
    subhaloFields = ['SubhaloFlag',
                     'SubhaloPos',
                     'SubhaloHalfmassRad',
                     'SubhaloHalfmassRadType',
                     'SubhaloLenType']
    
    #load subhalo catalog
    subhalos=il.groupcat.loadSubhalos(basePath,snapshot_number,fields=subhaloFields)  
    subhalos.keys()
    
    #get subhalo offset file for matching particle and subhalo IDs
    with h5py.File(offsetFile,'r') as f:
        SnapOffsetsSubhalo= np.copy(f['/Subhalo/SnapByType'])
    
    #get subhalo lengths for all gas particles
    SubhaloLenType = np.copy(subhalos['SubhaloLenType'])
    
    #create array of subhaloIDs for every gas particle
    AllShIDList = pID2shID(to_ID,partType,SubhaloLenType,SnapOffsetsSubhalo)
    
    #####################
    #SAVE LISTS TO FILES#
    #####################
    
    np.save('/{0}/PartList_Snap{1}.npy'.format(subdir_name,snapshot_number),AllPartList)
    np.save('/{0}/ShIDList_Snap{1}.npy'.format(subdir_name,snapshot_number),AllShIDList)    