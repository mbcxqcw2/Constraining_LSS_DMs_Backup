#######
#Notes#
#######

#This is a script for use as a batch job.
#It is based on ravemn:/u/cwalker/Illustris_Zhang_Method/WHIM_Pipes_Subhalo_Analysis_Parallel.ipynb
#It creates subhalo IDs for Pipe data which was created by make_pipes_5.py and stored with placeholder data.

#########
#imports#
#########

import os
import sys
import random

import numpy as np
import operator as op
import illustris_python as il

from frb.dm import igm
from charlie_TNG_tools import temp2u
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import binned_statistic_dd
from astropy.modeling import models, fitting

import multiprocessing as m
from contextlib import closing
from multiprocessing import Pool


from astropy import units as u
from numpy import random as rand
from astropy import constants as c
from matplotlib import pyplot as plt
from astropy.cosmology import Planck15 as cosmosource

###########
#Functions#
###########

#function to get true subhalo id information (from make_pipes_5.py)


def p2s_new(pipe_pIDs,chunk_pIDs,chunk_sIDs):
    """
    DESCRIPTION:
    
    Converts particle IDs to subhalo IDs. Specifically, this function:
    
    - Takes a pipe's particle IDs,
    - and takes a chunk of a simulation's particle ID and subhalo ID matchlists,
    - then outputs a list of subhalo IDs for the pipe.
    - if any pipe particle IDs appeared in the chunk,
    - the pipe's subhalo ID list will contain the corresponding value.
    - otherwise the subhalo ID will be -1 for that particle ID.
    
    Important note: Every matchlist chunk for the simulation must be looped over and 
    the results must be combined to have a comprehensive conversion for an entire pipe!
    
    NOTES:
    
    -Matchlists should have been created using Create_Particle_Subhalo_Matchlists_5.ipynb
    
    INPUTS:
    
    pipe_pIDs : [array-like] list of particle IDs in the pipe
    chunk_pIDs : [array-like] chunk of the particle-ID matchlist for a given simulation
    chunk_sIDs : [array-like] equivalent chunk of the subhalo-ID matchlist for the simulation
    
    RETURNS:
    
    pipe_sIDs : [array-like] list of subhalo IDs for the pipe, updated for any particles in
                the chunk. Any particles not in the chunk will be assigned a -1 subhalo ID.
                Every matchlist chunk for the simulation must be looped over and results
                must be combined to have a comprehensive conversion for an entire pipe!
    """
    
    #get unique particle IDs in pipe, and also inverse array and counts for each unique element
    pipe_p_uniques,pipe_p_inverse,pipe_p_counts = np.unique(pipe_pIDs,return_inverse=True,return_counts=True)
    
    #initialise a unique subhalo ID list for pipe
    pipe_s_uniques=np.ones_like(pipe_p_uniques)*-1 #initialise pipe shIDs
    
    ##find which pipe pIDs are in the chunk (NOTE: unnecessary step)
    #mask_1 = np.isin(pipe_p_uniques,chunk_pIDs)
    
    #find which chunk IDs are in the pipe
    mask_2=np.isin(chunk_pIDs,pipe_p_uniques)
    
    #create dictionary which can convert unique pIDs to their sIDs
    z = dict(zip(chunk_pIDs[mask_2],chunk_sIDs[mask_2]))
    
    #do the conversion for the uniques
    for i in range(len(pipe_p_uniques)):
        try:
            pipe_s_uniques[i]=z[pipe_p_uniques[i]]
        except:
            pass
        
    #recreate the non-uniques version of pIDs with the inverse array and counts
    #recreated_pipe_p = pipe_p_uniques[pipe_p_inverse]
    pipe_sIDs = pipe_s_uniques[pipe_p_inverse]
    
    return pipe_sIDs

def unwrap_package(package):
    """
    Helper function for parallel processing.
    Unpacks the set of data necessary for converting particle IDs
    to subhalo IDs for pipes created by make_pipes_5.py using
    process_pipe_batch().
    
    INPUTS:
    
    package : a list containing the input data, which are X rguments in the
              following order:
    
    all_pipes_pIDs   : [dict] data dictionary containing the particle
                       IDs of pipes which were created for a given simulation
                       and snapshot using make_pipes_5.py.
                       The dictionary should be loaded using:
                       
                       >np.load(file_path,allow_pickle=True).tolist()
                       
    file_path        : [str] location of the file which was loaded
                       as a data dictionary
                       
    simulation_id    : [str] the simulation which the data dictionary refers to
                       (e.g.) TNG300-1
                       
    snapshot_no      : [int] the snapshot number which the data dictionary
                       refers to (e.g.) 99
                       
    pID_ChunkList    : [list of str] list of locations of chunks of the particle
                       ID matchlist for this simulation/snapshot. These matchlist
                       chunks must have been created using:
                       
                       >Create_Particle_Subhalo_Matchlists_5.ipynb
                       
    sID_ChunkList    : [list of str] list of locations of chunks of the subhalo
                       ID matchlist for this simulation/snapshot. These matchlist
                       chunks must have been created using:
                       
                       >Create_Particle_Subhalo_Matchlists_5.ipynb
                       
                       
    pipes_to_process : [list] a list of ints. Each integer is a pipe from
                       all_pipes_data which will be processed
                       
    RETURNS:
    
    output of process_pipe_batch()
    
    """
    
    all_pipes_pIDs   = package[0]
    file_path        = package[1]
    simulation_id    = package[2]
    snapshot_no      = package[3]
    pID_ChunkList    = package[4]
    sID_ChunkList    = package[5]
    pipes_to_process = package[6]
    print('On this cpu will process pipes: {0}'.format(pipes_to_process))
    
    return process_pipe_batch(all_pipes_pIDs,
                              file_path,
                              simulation_id,
                              snapshot_no,
                              pID_ChunkList,
                              sID_ChunkList,
                              pipes_to_process)

def process_pipe_batch(all_pipes_pIDs,
                       file_path,
                       simulation_id,
                       snapshot_no,
                       pID_ChunkList,
                       sID_ChunkList,
                       pipes_to_process):
    """
    Processes a batch of pipes using p2s_new().
    
    INPUTS:
    
    all_pipes_pIDs   : [dict] data dictionary containing the particle
                       IDs of pipes which were created for a given simulation
                       and snapshot using make_pipes_5.py.
                       The dictionary should be loaded using:
                       
                       >np.load(file_path,allow_pickle=True).tolist()
                       
    file_path        : [str] location of the file which was loaded
                       as a data dictionary
                       
    simulation_id    : [str] the simulation which the data dictionary refers to
                       (e.g.) TNG300-1
                       
    snapshot_no      : [int] the snapshot number which the data dictionary
                       refers to (e.g.) 99
                       
    pID_ChunkList    : [list of str] list of locations of chunks of the particle
                       ID matchlist for this simulation/snapshot. These matchlist
                       chunks must have been created using:
                       
                       >Create_Particle_Subhalo_Matchlists_5.ipynb
                       
    sID_ChunkList    : [list of str] list of locations of chunks of the subhalo
                       ID matchlist for this simulation/snapshot. These matchlist
                       chunks must have been created using:
                       
                       >Create_Particle_Subhalo_Matchlists_5.ipynb
                       
                       
    pipes_to_process : [list] a list of ints. Each integer is a pipe from
                       all_pipes_data which will be processed
    
    RETURNS:
    
    None, but saves data
    """

    
    #loop over pipes to process
    for i in range(len(pipes_to_process)):
        
        #select which pipe to process from list of all pipes
        pipe_to_process = pipes_to_process[i]
        print('processing pipe {0}...'.format(pipe_to_process))
    
        #output file name for this pipe
        outfilename='{0}/sim_{1}_snap_{2}_pipe_{3}_true_shID_list.npy'.format(file_path,
                                                                              simulation_id,
                                                                              snapshot_no,
                                                                              pipe_to_process)
        
        #if file containing subhalo IDs for this pipe exists already...
        if os.path.isfile(outfilename)==True:
            print('...pipe has already been processed.') #skip processing it again

        #else process this pipe
        else:
            
            #extract particle IDs for the chosen pipe
            pipe_pIDs = np.array(all_pipes_pIDs[pipe_to_process])
            
            #initialise subhalo ID list for pipe
            pipe_sIDs = np.ones_like(pipe_pIDs)*-1
            
            #loop over each chunk of the matchlists
            for j in range(len(pID_ChunkList)):
                print('Chunk {0}'.format(j))
                
                #load pID/shID chunk to search
                chunk_pIDs = np.load(pID_ChunkList[j])
                chunk_sIDs = np.load(sID_ChunkList[j])
                #print(chunk_pIDs.shape,chunk_sIDs.shape)
                
                #get subhalo ID list which has been updated for only this chunk
                out_ = p2s_new(pipe_pIDs,chunk_pIDs,chunk_sIDs)
                
                #update whole subhalo ID list with this chunk's updates
                inds = np.where(out_!=-1)
                pipe_sIDs[inds] = out_[inds]
        
            #once all chunks are processed, save the output subhalo ID list
            #as a numpy array
            np.save(outfilename,pipe_sIDs)

    
    return

##############################
#parse command line arguments#
##############################

cla = sys.argv
#cla = ['','TNG300-1','99','4'] #fake command line input


n_inputs = 3

if len(cla)!=n_inputs+1:
    print("Error! {0} arguments required. {1} arguments provided.".format(n_inputs,len(cla)))
    print("Exiting script.")
    sys.exit()

else:
    
    sim_to_load = str(cla[1])# the simulation to load data for, e.g. 'TNG300-1'
    snap_to_process = int(cla[2]) #the snapshot number to process data for, e.g. 99
    cpus_to_use = int(cla[3]) #the number of simultaneous cores to load data with.
    
    print('Command line inputs: sim={0}, snap={1}, cpus={2}'.format(sim_to_load,
                                                                    snap_to_process,
                                                                    cpus_to_use))

############
#Initialise#
############


#path to simulation data
basePath = '/ptmp/cwalker/Illustris_FRB_Project/TNG_copies/virgo/simulations/IllustrisTNG/{0}/output/'.format(sim_to_load)


#base directory containing pipe files for data with WHIM info
pipeBasePath = '/u/cwalker/Illustris_Zhang_Method/SpeedTempTest/'

#toggle parallel processing on/off
parallel_code_test = True


#############################
#Identify files to be loaded#
#############################

#load the files from the base path
all_files = os.listdir(pipeBasePath)#for pipes created for smaller simulations

#select all .npy files in directory
npy_files = [i for i in all_files if '.npy' in i]

#select all ddmdz (i.e. pipe) files in directory
dDMdz_files = [i for i in npy_files if 'dDMdz_Output' in i]

#select files of correct simulation
dDMdz_files = [i for i in npy_files if sim_to_load in i]

#make sure name fits the WHIM output files
dDMdz_files = [i for i in dDMdz_files if 'SpeedTempTest' in i]
        
#create a list of snapshots containing pipes in this directory
snap_list = [int(i.split('_')[3]) for i in dDMdz_files]

#sort list into ascending order (i.e. high to low redshift)
dDMdz_files = [x for _, x in sorted(zip(snap_list, dDMdz_files))]
snap_list.sort()

#reverse list into descending order (i.e. low to high redshift)
dDMdz_files = dDMdz_files[::-1]
snap_list = snap_list[::-1]

print(snap_list)

#just process desired snapshot
snap_list = [i for i in snap_list if snap_to_process == i]
dDMdz_files = [i for i in dDMdz_files if '{0}'.format(snap_to_process) in i]

#print to check which files will be processed.
print(snap_list)
print(dDMdz_files)

################################################
#load file containing pipes for chosen sim/snap#
################################################

snapshot = str(snap_list[0]) #string version of snapshot to load data for
print('Loading snapshot: {0}'.format(snapshot))

file_path = pipeBasePath+dDMdz_files[0] #full path to data file
print('Loading data at: {0}'.format(file_path))

vals = np.load(file_path,allow_pickle=True).tolist() #extract values from file
print('Data loaded.')

#explore what data exists for a file
print('Keys in this file: {0}'.format(vals.keys()))

#explore the particle id key which will be used to create subhalo ids
print('number of sightlines in file: {0}'.format(len(np.array(vals['dDMdz_Pakmor']))))
print('number of Particle ID lists in file: {0}'.format(len(np.array(vals['LoSPartIDs']))))

########################################################
#get matchlist chunks which will be used for conversion#
########################################################

#get list of matchlist chunks
print('Using new version of code which loads particle/subhalo matchlists in chunks from /ptmp/')
Chunked_loc = '/ptmp/cwalker/Illustris_FRB_Project/Sim_Matchlists/Matchlist_dir_{0}/'.format(sim_to_load) #location of the chunked data

#get the particle ID list chunks
ChunkedPartIDs = os.listdir(Chunked_loc)
ChunkedPartIDs = ['{0}/{1}'.format(Chunked_loc,i) for i in ChunkedPartIDs if 'PartList_Snap{0}_Chunk'.format(snapshot) in i]
ChunkedPartIDs.sort()

#get the subhalo ID list chunks
ChunkedSubhIDs = os.listdir(Chunked_loc)
ChunkedSubhIDs = ['{0}/{1}'.format(Chunked_loc,i) for i in ChunkedSubhIDs if 'ShIDList_Snap{0}_Chunk'.format(snapshot) in i]
ChunkedSubhIDs.sort()
    
print(ChunkedPartIDs,ChunkedSubhIDs)

############################################
############################################
##The parallel processing part of the code##
############################################
############################################

#############################
#create true subhalo ID list#
#############################

#get the number of created pipes for the snapshot
n_pipes_to_process = np.array(vals['LoSPartIDs']).shape[0]
print('number of pipes to process (should be 5125): {0}'.format(n_pipes_to_process))

#get number of bins per pipe
n_bins_per_pipe = np.array(vals['LoSPartIDs'])[0,:].shape
print('number of bins per pipe (should be 10,000): {0}'.format(n_bins_per_pipe))

################################
#Case 1: No Parallel Processing#
################################

#if subhalos will be analysed one by one...
if parallel_code_test == False:
    
    print('Will not process in parallel.')

    #test for only first pipe
    pipe_pIDs = np.array(vals['LoSPartIDs'][0])

    #initialise subhalo ID list for pipe
    pipe_sIDs = np.ones_like(pipe_pIDs)*-1

    #loop over each chunk of the matchlist
    for j in range(len(ChunkedPartIDs)):#[298,299,300,301,302]:
        print(j)

        #load pID/shID chunk to search
        chunk_pIDs = np.load(ChunkedPartIDs[j])
        chunk_sIDs = np.load(ChunkedSubhIDs[j])
        print(chunk_pIDs.shape,chunk_sIDs.shape)

        #get sh ID list updated for THIS CHUNK ONLY
        #they need to be combined somehow during the loop over chunks
        out_ = p2s_new(pipe_pIDs,chunk_pIDs,chunk_sIDs)

        #update whole subhalo ID list with this chunk's updates
        inds = np.where(out_!=-1)
        pipe_sIDs[inds] = out_[inds]

        print(pipe_sIDs)
        print(np.where(pipe_sIDs!=-1))

#############################
#Case 2: Parallel Processing#
#############################
        
#else if subhalos will be analysed in parallel:
elif parallel_code_test == True:
    
    ##############################
    #specifics of parallelisation#
    ##############################
    
    print('Will process in parallel.')
    
    n_full_core = n_pipes_to_process//cpus_to_use
    print('Will use {0} out of a maximum of {1} cpus in parallel'.format(n_full_core,m.cpu_count()))
    print('First {0} cpus will process {1} pipes each in parallel.'.format(cpus_to_use,n_full_core))
    
    n_partial_core = n_pipes_to_process%cpus_to_use
    print('Then will process remaining {0} pipes on a single cpu'.format(n_partial_core))
    
    ############################
    #do the parallel processing#
    ############################
    
    ###################################
    #scenario 1: there is no remainder#
    ###################################
    
    if n_partial_core == 0:
        
        #create the cpu map
        cpu_map = np.arange(n_full_core*cpus_to_use).reshape(cpus_to_use,n_full_core)
        print('CPU map: {0}'.format(cpu_map))
        
        #create the package to unwrap during multiprocessing
        package = [(vals['LoSPartIDs'],
                    pipeBasePath,
                    sim_to_load,
                    snap_to_process,
                    ChunkedPartIDs,
                    ChunkedSubhIDs,
                    cpu_map[i]) for i in range(cpus_to_use)]
        
        #process
        with closing(Pool(cpus_to_use)) as p: #invoke multiproccessing
            run = p.map(unwrap_package,package,chunksize=1) #run the multiprocessing
        p.terminate() #terminate after completion
    
    ################################
    #scenario 2: there is remainder#
    ################################
    
    elif n_partial_core !=0:
        
        #create the cpu maps
        cpu_map_a = np.arange(n_full_core*cpus_to_use).reshape(cpus_to_use,n_full_core) #full core map
        cpu_map_b = np.arange(n_full_core*cpus_to_use,n_pipes_to_process).reshape(n_partial_core,1) #partial core map
        print('CPU maps:\n Full Core Map: {0}\n Partial Core Map: {1}'.format(cpu_map_a,cpu_map_b))
        
        #create the packages to unwrap during multiprocessing
        package_a = [(vals['LoSPartIDs'],
                      pipeBasePath,
                      sim_to_load,
                      snap_to_process,
                      ChunkedPartIDs,
                      ChunkedSubhIDs,
                      cpu_map_a[i]) for i in range(cpus_to_use)]
        package_b = [(vals['LoSPartIDs'],
                      pipeBasePath,
                      sim_to_load,
                      snap_to_process,
                      ChunkedPartIDs,
                      ChunkedSubhIDs,
                      cpu_map_b[i]) for i in range(n_partial_core)]
        
        #full core multiprocessing
        print('full core')
        with closing(Pool(cpus_to_use)) as p: #invoke multiproccessing
            run = p.map(unwrap_package,package_a,chunksize=1) #run the multiprocessing
        p.terminate() #terminate after completion
        
        #partial core multiprocessing
        print('partial core')
        with closing(Pool(n_partial_core)) as p: #invoke multiproccessing
            run = p.map(unwrap_package,package_b,chunksize=1) #run the multiprocessing
        p.terminate() #terminate after completion  