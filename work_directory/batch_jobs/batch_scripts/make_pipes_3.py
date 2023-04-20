#Notes
#This is a script for use as a batch job
#It is an upgade of make_pipes_2.py.
#It incorporates updated code for impact factor analysis
#It incorporates code which allows subhalo/particle ID matchlists to be loaded in chunks when stored in /ptmp/
#and multiprocessing from ../Illustris_Zhang_Method/Pipe_Creation_Plus_LSS_9.ipynb

#imports

import os
import sys
import h5py

import numpy as np
import multiprocessing as m
import illustris_python as il

from contextlib import closing
from multiprocessing import Pool

from numpy import random as rand
from astropy import constants as c
from charlie_TNG_tools import temp2u
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import binned_statistic_dd
from astropy.cosmology import Planck15 as cosmosource

#parse command line arguments#
cla = sys.argv
n_inputs = 4

if len(cla)!=n_inputs+1:
    print("Error! {0} arguments required. {1} arguments provided.".format(n_inputs,len(cla)))
    print("Exiting script.")
    sys.exit()

else:
    sim_to_use = str(cla[1])# the simulation to use, e.g. 'TNG50-4'
    pipes_per_snap = int(cla[2]) #the number of pipes to create per snapshot, e.g. 5125
    snap_to_process = int(cla[3]) #the snapshot number to process, e.g. 99
    cpus_to_use = int(cla[4]) #the number of simultaneous cores to load data with.

#Functions

def gadgetDens2SI(dens):
    """
    Original Artale function to convert TNG densities to SI units

    INPUTSRETURNS:

    dens : [values] densities from TNG

    RETURNS:

    dens converted to SI units
    """
    return dens*1E10*cel_Msol_si/cel_hubble/(cel_kpc_si/cel_hubble)**3

def TNG_Dens2SI(dens):
    """
    Like gadgetDens2SI but using astropy values for constants
    Strips result of units
    Developed in raven:/u/cwalker/Illustris_FRB_Project/yt-artale-constants.ipynb


    INPUTSRETURNS:

    dens : [values] densities from TNG

    RETURNS:

    dens converted to SI units
    """
    return dens*1E10*c.M_sun.to('kg').value/cosmosource.h/(c.kpc.to('m').value/cosmosource.h)**3

def TNG_Dens2SI_astropy(dens):
    """
    Like TNG_Dens2SI but does not strip result of units.
    Developed in raven:/u/cwalker/Illustris_FRB_Project/yt-artale-constants.ipynb


    INPUTSRETURNS:

    dens : [values] densities from TNG

    RETURNS:

    dens converted to SI units
    """

    return dens*1E10*c.M_sun.to('kg')/cosmosource.h/(c.kpc.to('m')/cosmosource.h)**3


def pSplitRange(indrange, numProcs, curProc, inclusive=False):
    """ Divide work for embarassingly parallel problems. 
    Accept a 2-tuple of [start,end] indices and return a new range subset.
    If inclusive==True, then assume the range subset will be used e.g. as input to snapshotSubseet(),
    which unlike numpy convention is inclusive in the indices."""
    assert len(indrange) == 2 and indrange[1] > indrange[0]

    if numProcs == 1:
        if curProc != 0:
            raise Exception("Only a single processor but requested curProc>0.")
        return indrange

    # split array into numProcs segments, and return the curProc'th segment
    splitSize = int(np.floor( (indrange[1]-indrange[0]) / numProcs ))
    start = indrange[0] + curProc*splitSize
    end   = indrange[0] + (curProc+1)*splitSize

    # for last split, make sure it takes any leftovers
    if curProc == numProcs-1:
        end = indrange[1]

    if inclusive and curProc < numProcs-1:
        # not for last split/final index, because this should be e.g. NumPart[0]-1 already
        end -= 1

    return [start,end]


def loadSubset(simPath, snap, partType, fields, chunkNum=0, totNumChunks=1):
    """ 
    Load part of a snapshot.
    frm Dylan Nelson: https://www.tng-project.org/data/forum/topic/203/loading-the-tng100-1-data/
    """
    nTypes = 6
    ptNum = il.util.partTypeNum(partType)

    with h5py.File(il.snapshot.snapPath(simPath,snap),'r') as f:
        numPartTot = il.snapshot.getNumPart( dict(f['Header'].attrs.items()) )[ptNum]

    # define index range
    indRange_fullSnap = [0,numPartTot-1]
    indRange = pSplitRange(indRange_fullSnap, totNumChunks, chunkNum, inclusive=True)

    # load a contiguous chunk by making a subset specification in analogy to the group ordered loads
    subset = { 'offsetType'  : np.zeros(nTypes, dtype='int64'),
               'lenType'     : np.zeros(nTypes, dtype='int64') }

    subset['offsetType'][ptNum] = indRange[0]
    subset['lenType'][ptNum]    = indRange[1]-indRange[0]+1

    # add snap offsets (as required)
    with h5py.File(il.snapshot.offsetPath(simPath,snap),'r') as f:
        subset['snapOffsets'] = np.transpose(f['FileOffsets/SnapByType'][()])

    # load from disk
    r = il.snapshot.loadSubset(simPath, snap, partType, fields, subset=subset)

    return r

def process_sim_chunk(snap_number,basePath,sim_to_use,nSubLoads,chunkIDs,T_h,T_c,c1s,c2e):
    """
    processes part of the simulation on a single cpu. Is fed by unwrap_package().
    this is specifically for pipes going along the x-axis from 0 to box length.
    
    
    INPUTS:
    
    snap_number : [int] the snapshot number of the simulation to be processed
    basePath    : [str] the path to the simulation data to be processed
    sim_to_use  : [str] the simulation to be processed
    nSubLoads   : [int] the total number of parts the simulation will be split into
    chunkIDs    : [array of ints] the id numbers of the chunks of simulation to be processed on this cpu
    T_h         : [= 10**7]  hot phase gase temperature [Kelvin] 
    T_c         : [= 10**3]  cold phase gas temperature [Kelvin]
    c1s         : [0,pipe_width/2,pipe_width/2] coordinates at upper right of pipe start
    c2e         : [0,-pipe_width/2,-pipe_width/2] coordinates at lower left of pipe end
    
    
    RETURNS:
    
    """
    
    verbose = False
    
    for i in range(len(chunkIDs)): #loop over all chunk ids
        
        chunkID = chunkIDs[i] # the chunk ID to be processed
        
        if verbose == True:
            print('Verbose mode check: chunkID = {0}'.format(chunkID))
        
        temp_dict = {} #initialise dictionary to store data for this chunk in
        
        #the name of the file dictionary for this chunk will be stored in
        part_outdata_filename = '/u/cwalker/Illustris_Zhang_Method/temp_chunks/sim_{0:02d}_snap_{1:03d}_cID_{2}.npy'.format(snap_number,chunkID,sim_to_use) 
        
        if verbose == True:
            print('Verbose mode check: store data at = {0}'.format(part_outdata_filename))
            
            
        #initialise arrays to hold the desired information about
        #the simulation cells in this part of the simulation

        #pipe_cell_coords_part=[]
        #pipe_cell_dens_part = []
        #pipe_cell_elab_part=[]
        #pipe_cell_sfr_part=[]
        #pipe_cell_dark_part = []
        #pipe_cell_warm_part=[]
        #pipe_cell_pIDs_part=[]

        ###########################
        #load the partial data set#
        ###########################

        data = loadSubset(basePath,snap_number, 'gas', fields,chunkNum=chunkID, totNumChunks=nSubLoads)
        
        if verbose == True:
            print('Verbose mode check: data = {0}'.format(data))

        #####################################
        #create warm-phase gas mass fraction#
        #####################################

        density = data['Density'] #the density values along the light ray in gcm**-3
        sfr     = data['StarFormationRate'] #the star formation rate along the light ray in g/s
        ie      = data['InternalEnergy'] #the internal energy along the light ray in erg/g
        ea      = data['ElectronAbundance'] #the electron abundance along the light ray
        #calculate x and w, cold and warm phase gas mass fractions
        x_frac = (temp2u(T_h,ea)-ie)/(temp2u(T_h,ea)-temp2u(T_c,ea)) #cold phase mass fraction
        w_frac = 1 - x_frac # warm phase mass fraction
        #only modify electron abundance if sfr = 0
        w_frac[np.where(sfr==0)]=1
        data['Warm']=w_frac    
        
        if verbose == True:
            print('Verbose mode check: w frac = {0}'.format(data['Warm']))

        ########################
        #get cells in this pipe#
        ########################

        yz_pts = data['Coordinates'][:,[1,2]]
        ur = c1s[1:] #upper right of pipe start (y and z only)
        ll = c2e[1:] #lower left of pipe end (y and z only)
        inidx = np.all((ll <= yz_pts) & (yz_pts <= ur), axis=1) #indexes of cells in pipe
        
        if verbose == True:
            print('Verbose mode check: inidx = {0}'.format(inidx))

        ###########################
        #get data of cells in pipe#
        ###########################

        #pipe_cell_coords_part.append(data['Coordinates'][inidx])       #coordinates [ckpc/h]
        pipe_cell_coords_part=data['Coordinates'][inidx]      #coordinates [ckpc/h]

        #pipe_cell_dens_part.append(data['Density'][inidx])           #densities [(1e10Msun/h)/(ckpc/h)**3]
        pipe_cell_dens_part=data['Density'][inidx]          #densities [(1e10Msun/h)/(ckpc/h)**3]

        #pipe_cell_elab_part.append(data['ElectronAbundance'][inidx]) #electron abundance [-]
        pipe_cell_elab_part=data['ElectronAbundance'][inidx] #electron abundance [-]

        #pipe_cell_sfr_part.append(data['StarFormationRate'][inidx]) #star formation rate [Msun/yr]
        pipe_cell_sfr_part=data['StarFormationRate'][inidx] #star formation rate [Msun/yr]

        #pipe_cell_dark_part.append(data['SubfindDMDensity'][inidx])  #comoving dark matter density [(1e10Msun/h)/(ckpc/h)**3]
        pipe_cell_dark_part=data['SubfindDMDensity'][inidx]  #comoving dark matter density [(1e10Msun/h)/(ckpc/h)**3]

        #pipe_cell_warm_part.append(data['Warm'][inidx])
        pipe_cell_warm_part=data['Warm'][inidx]

        #pipe_cell_pIDs_part.append(data['ParticleIDs'][inidx])
        pipe_cell_pIDs_part=data['ParticleIDs'][inidx]
        
        #########################################
        #store these to a dictionary to be saved#
        #########################################
        
        temp_dict['Coordinates']       = pipe_cell_coords_part
        temp_dict['Density']           = pipe_cell_dens_part
        temp_dict['ElectronAbundance'] = pipe_cell_elab_part
        temp_dict['StarFormationRate'] = pipe_cell_sfr_part
        temp_dict['SubfindDMdensity']  = pipe_cell_dark_part
        temp_dict['Warm']              = pipe_cell_warm_part
        temp_dict['ParticleIDs']       = pipe_cell_pIDs_part
        
        if verbose == True:
            print('Verbose mode check: temp dict = {0}'.format(temp_dict))

        #####################################################################
        #save data to temporary array for loading with the rest of the parts#
        #####################################################################
        np.save('{0}'.format(part_outdata_filename),temp_dict)

        if verbose == True:
            print('Verbose mode check: saved')
    
    
    return

def unwrap_package(package):
    """
    Helper function for parsing simulation in parallel using multiprocessing.
    Unpacks the set of data necessary for parsing the simulation.
    Then parses the simulation using process_sim_chunk().
    
    INPUTS:
    
    package : a list containing the input data, which are X arguments in the following order:
    
        snap_number : [int] the snapshot number of the simulation to be processed
        basePath    : [str] the path to the simulation data to be processed
        sim_to_use  : [str] the simulation to be processed
        nSubLoads   : [int] the total number of parts the simulation will be split into
        chunkIDs    : [array of ints] the id numbers of the chunks of simulation to be processed on this cpu
        T_h         : [= 10**7]  hot phase gase temperature [Kelvin] 
        T_c         : [= 10**3]  cold phase gas temperature [Kelvin]
        c1s         : [0,pipe_width/2,pipe_width/2] coordinates at upper right of pipe start
        c2e         : [0,-pipe_width/2,-pipe_width/2] coordinates at lower left of pipe end
    
    
    RETURNS:
    
    output of process_package()
    """
    
    verbose=False
    
    #unwrap the package to feed to process_sim_chunk()
    snap_number = package[0]
    if verbose==True:
        print('Verbose mode check: snap_number = {0}'.format(snap_number))
    basePath    = package[1]
    if verbose==True:
        print('Verbose mode check: basePath = {0}'.format(basePath))
    sim_to_use  = package[2]
    if verbose==True:
        print('Verbose mode check: sim_to_use = {0}'.format(sim_to_use))
    nSubLoads   = package[3]
    if verbose==True:
        print('Verbose mode check: nSubLoads = {0}'.format(nSubLoads))
    chunkIDs    = package[4]
    if verbose==True:
        print('Verbose mode check: chunkIDs = {0}'.format(chunkIDs))
    T_h         = package[5]
    if verbose==True:
        print('Verbose mode check: T_h = {0}'.format(T_h))
    T_c         = package[6]
    if verbose==True:
        print('Verbose mode check: T_c = {0}'.format(T_c))
    c1s         = package[7]
    if verbose==True:
        print('Verbose mode check: c1s = {0}'.format(c1s))
    c2e         = package[8]
    if verbose==True:
        print('Verbose mode check: c2e = {0}'.format(c2e))
    
    print('torun: ',snap_number,basePath,sim_to_use,nSubLoads,chunkIDs,T_h,T_c,c1s,c2e)
    
    
    #run process_sim_chunk()
    process_sim_chunk(snap_number,basePath,sim_to_use,nSubLoads,chunkIDs,T_h,T_c,c1s,c2e)
    
    return 'done'

def oldpIDshIDconverter(pipe_cell_pIDs,AllPartIDs,AllSubhIDs):
    """
    Old version of the code which took a set of particle IDs and created a list of 
    corresponding subhalo IDs
    
    INPUTS:
    
    pipe_cell_pIDs : the particle ids of cells in a given pipe
    AllPartIDs     : particle ID list for all cells in the desired simulation
    AllSubhIDs     : subhalo ID list for all cells in the desired simulation
    
    RETURNS:
    
    pipe_cell_shIDs : the corresponding subhalo IDs for every particle id in the cell.
    
    
    
    """

    #print('Conversion check')

    #create a set of particle IDs for the cells in this pipe
    PartID_Set = set(pipe_cell_pIDs.tolist())

    #initialise an array to contain the corresponding positions within the simulation of these cells
    sim_inds = np.zeros(pipe_cell_pIDs.shape,dtype=int)

    #loop over all particle IDs in the desired simulation
    for i, x in enumerate(AllPartIDs):

        #find when particle ID is also in the pipe
        if x in PartID_Set:

            #find where that particle ID is in the pipe
            pipe_idx = np.where(pipe_cell_pIDs==x)

            #assign the pipe at that point the cell's corresponding simulation position
            sim_inds[pipe_idx] = i


    #for all of these simulation positions, get the correct subhalo ID
    pipe_cell_shIDs = np.array(AllSubhIDs[sim_inds])
    #print(pipe_cell_shIDs)

    #print('Conversion check end')

    return pipe_cell_shIDs

def newpIDshIDconverter(pipe_cell_pIDs,ChunkedPartIDs,ChunkedSubhIDs):
    """
    New version of the code which creates a set of subhalo IDs from a set of particle
    IDs.
    
    This version loops through each chunk of the simulation ID lists in turn searching
    for relevant particle and subhalo IDs. 
    
    Note: could be improved to be faster if, when all correct particle IDs are found, it
    does not need to search further chunks. This is not yet implemented.
    
    INPUTS:
    
    pipe_cell_pIDs : the particle ids of cells in a given pipe
    ChunkedPartIDs : list containing locations of the chunks of 
                     the particle ID list for all cells in the 
                     desired simulation. If all of these were loaded
                     into a single array, the result would be the
                     same as AllPartIDs in oldpIDshIDconverter().
    ChunkedSubhIDs : list containing locations of the chunks of 
                     the subhalo ID list for all cells in the 
                     desired simulation. If all of these were loaded
                     into a single array, the result would be the
                     same as AllPartIDs in oldpIDshIDconverter().
    
    RETURNS:
    
    pipe_cell_shIDs : the corresponding subhalo IDs for every particle id in the cell.

    
    """
    
    #create a set of particle IDs for the cells in this pipe
    PartID_Set = set(pipe_cell_pIDs.tolist())
    
    #initialise an array to contain all subhalo IDs in simulation
    pipe_cell_shIDs = np.ones(pipe_cell_pIDs.shape,dtype=int)*-1
    
    #print(ChunkedPartIDs)
    
    #load chunks of the all-simulation particle ID list
    for i in range(len(ChunkedPartIDs)):
        
        #True/False array of same shape as pipe data which allows 
        #us to extract the relevant IDs from each chunk
        TF_arr = np.full(pipe_cell_pIDs.shape, False) #begin with false, flip to true when chunk contains ID
        
        sim_inds = []#initialise an array to contain  positions of any cells in this chunk
        
        #get location of chunks to load
        PartFile_toload = ChunkedPartIDs[i] #all-simulation particle ID list chunk
        SubhFile_toload = ChunkedSubhIDs[i] #all-simulation subhalo ID list chunk
        
        #load the ID list chunks
        ChunkOfPartIDs = np.load(PartFile_toload) #particle chunk
        ChunkOfSubhIDs = np.load(SubhFile_toload) #subhalo chunk

        #loop over the particle IDs in the chunk
        for j, x in enumerate(ChunkOfPartIDs):
            
            #find if particle ID is also in the pipe
            if x in PartID_Set:
                                
                #find where that particle ID is in the pipe
                pipe_idx = np.where(pipe_cell_pIDs==x)
                
                #flip the True/False array index to True for this cell
                TF_arr[pipe_idx] = True
                
                #append the cell's corresponding chunk position
                sim_inds.append(j)
                
                #print(i,j,pipe_idx,x,ChunkOfSubhIDs[j])

            
        #convert all chunk position indices to array
        sim_inds = np.array(sim_inds)
        
        #record all corresponding subhalo IDs in this chunk
        #print(pipe_cell_shIDs[TF_arr])
        #print(sim_inds)
        #print(ChunkOfSubhIDs[sim_inds])
        if sim_inds.size>0:#only try this if the array is not empty
            pipe_cell_shIDs[TF_arr] = ChunkOfSubhIDs[sim_inds]
            

    return pipe_cell_shIDs

############
#initialise#
############

sim_to_use = sim_to_use
print('Simulation to use will be: {0}'.format(sim_to_use))

pipes_per_snap = pipes_per_snap
print('Number of pipes to create per snapshot: {0}'.format(pipes_per_snap))

snaps_to_process = [snap_to_process]#,13,11,8,6,4,3,2]
print('Snapshots to process will be {0}'.format(snaps_to_process))

#The number of cells in the chosen snapshot
#ncells = dataPT0['Coordinates'].shape[0]
#print('Number of cells in snapshot {0} is {1}'.format(snap_number,ncells))

#The width of the pipe
pipe_width = 200 #By following zhang+20 definition, sides will be 200ckpc/h in length
print('Pipe width will be {0} ckpc/h'.format(pipe_width))

#The number of bins along a single line of sight
nbins=10000 #Zhang+20 definition: 10,000
print('There will be {0} bins on each sightline'.format(nbins))

#Define the mass of a proton for dDM/dz calculations
protonmass = c.m_p.to('kg')
print('Proton mass is {0}'.format(protonmass))

#Define the hydrogen mass fraction for dDM/dz calculations
hmassfrac = 3./4.
print('Chosen H mass fraction is {0}. Check whether this is correct'.format(hmassfrac))

#calculate the critical density at redshift zero for structure categorisation
#source to formula: https://astronomy.swin.edu.au/cosmos/c/Critical+Density
grav=c.G.to('m**3/(kg*s**2)') #g as a YT quantity in correct units
H=cosmosource.H(0).to('km/s/Mpc') #hubble const at z=0 in km/s/Mpc
my_dens_crit = ((3 * H**2)/(8*np.pi* grav)).to('kg/m**3')
print('Critical density at z=0 = {0}'.format(my_dens_crit))

nSubLoads = 100 #number of subloads to split simulation into

##pipe info for test
#npipes      = 1  #number of pipes to create
#snap_number = 99 #snapshot number for test


#base path to simulation
#basePath = '/virgo/simulations/IllustrisTNG/{0}/output/'.format(sim_to_use)
basePath = '/ptmp/cwalker/Illustris_FRB_Project/TNG_copies/virgo/simulations/IllustrisTNG/{0}/output/'.format(sim_to_use)
#basePath = '/virgo/simulations/IllustrisTNG/{0}/output/'.format(sim_to_use)

#load header
#header = il.groupcat.loadHeader(basePath,snap_number)

#fields to load for test
fields=['Density',
        'ElectronAbundance',
        'StarFormationRate',
        'InternalEnergy',
        'Coordinates',
        'Masses',
        'SubfindDMDensity',
        'ParticleIDs'] 

#define constants foor warm-phase gas mass fraction calculation
T_h = 10**7  #hot phase gase temperature [Kelvin]
T_c = 10**3  #cold phase gas temperature [Kelvin]
x_h = 0.75   #Hydrogen mass fraction

#identify number of available cores on the system
ncpus = m.cpu_count()

#choose the number of cores to use at once. 
cpus_to_use = cpus_to_use 

#calculate the number of full core runs to be used to check for simulation cells in pipe
#this number is the number of parts of the simulation which will be loaded simultaneously
n_full_core = nSubLoads//cpus_to_use

#calculate the number of cores which must be used to check the remaining simulation cells
#this number is the number of leftover parts of the simulation which will be loaded all at once
n_partial_core = nSubLoads%cpus_to_use

print('To parse simulation data, {0} cpus will load data simultaneously. This will happen {1} times. The remaining data needs {2} cpus. These will be loaded simultaneously.'.format(cpus_to_use,n_full_core,n_partial_core))

#if statement to allow testing of whether multiproccessing-related functions are working correctly
#If it is set to False, multiprocessing is enabled.
#If set to true, everything is done sequentially with no multiprocessing.
parallelcodetest = False



########################
########################
##ACTUAL PIPE CREATION##
########################
########################

pIDshID_version = 'new' #a switch while testing the new vs old shID versions, can be old, new, both


#####################
#Loop over snapshots#
#####################

for snapshot_to_process in range(len(snaps_to_process)):
   
    ############
    #initialise#
    ############
    
    npipes            = pipes_per_snap  #number of pipes to create
    snap_number       = snaps_to_process[snapshot_to_process] #snapshot number for test
    
    print('Currently processing snapshot: {0}'.format(snap_number))
    
    
    #############
    #load header#
    #############
    
    header = il.groupcat.loadHeader(basePath,snap_number)
    print('Header for snap = {0}'.format(header))
    
    #####################################################################################################
    #edit for larger simulations: put this in an if statement to test both chunkeed and whole matchlists#
    #####################################################################################################
    
    #load the particle and subhalo ID lists for all cells in the desired simulation
    if pIDshID_version == 'old':
        #load the whole matchlists
        AllPartIDs = np.load('/u/cwalker/Illustris_Zhang_Method/Sim_Matchlists/Matchlist_dir_{0}/PartList_Snap{1}.npy'.format(sim_to_use,snap_number))
        AllSubhIDs = np.load('/u/cwalker/Illustris_Zhang_Method/Sim_Matchlists/Matchlist_dir_{0}/ShIDList_Snap{1}.npy'.format(sim_to_use,snap_number))
    
    elif pIDshID_version == 'new':
        #get list of matchlist chunks
        print('Using new version of code which loads particle/subhalo matchlists in chunks from /ptmp/')
        Chunked_loc = '/ptmp/cwalker/Illustris_FRB_Project/Sim_Matchlists/Matchlist_dir_{0}/'.format(sim_to_use) #location of the chunked data
        
        #get the particle ID list chunks
        ChunkedPartIDs = os.listdir(Chunked_loc)
        ChunkedPartIDs = ['{0}/{1}'.format(Chunked_loc,i) for i in ChunkedPartIDs if 'PartList_Snap{0}_Chunk'.format(snap_number) in i]
        ChunkedPartIDs.sort()
        
        #get the subhalo ID list chunks
        ChunkedSubhIDs = os.listdir(Chunked_loc)
        ChunkedSubhIDs = ['{0}/{1}'.format(Chunked_loc,i) for i in ChunkedSubhIDs if 'ShIDList_Snap{0}_Chunk'.format(snap_number) in i]
        ChunkedSubhIDs.sort()
        
    elif pIDshID_version == 'both':
        print('Checking output of new and old subhalo ID generation methods...')
        #load the whole matchlists
        AllPartIDs = np.load('/u/cwalker/Illustris_Zhang_Method/Sim_Matchlists/Matchlist_dir_{0}/PartList_Snap{1}.npy'.format(sim_to_use,snap_number))
        AllSubhIDs = np.load('/u/cwalker/Illustris_Zhang_Method/Sim_Matchlists/Matchlist_dir_{0}/ShIDList_Snap{1}.npy'.format(sim_to_use,snap_number))        

        #get list of matchlist chunks
        Chunked_loc = '/ptmp/cwalker/Illustris_FRB_Project/Sim_Matchlists/Matchlist_dir_{0}/'.format(sim_to_use) #location of the chunked data
        print('Location of Particle ID matchlist chunks: {0}'.format(Chunked_loc))
        
        #get the particle ID list chunks
        ChunkedPartIDs = os.listdir(Chunked_loc)
        #print('All files in the location: {0}'.format(ChunkedPartIDs))
        print('Testing for string: {0}'.format('PartList_Snap{0}_Chunk'.format(snap_number)))
        #print([i for i in ChunkedPartIDs if 'PartList_Snap{0}_Chunk'.format(snap_number) in i])
        ChunkedPartIDs = ['{0}/{1}'.format(Chunked_loc,i) for i in ChunkedPartIDs if 'PartList_Snap{0}_Chunk'.format(snap_number) in i]
        ChunkedPartIDs.sort()
        #print('Particle ID matchlist chunks: {0}'.format(ChunkedPartIDs))
        
        #get the subhalo ID list chunks
        ChunkedSubhIDs = os.listdir(Chunked_loc)
        ChunkedSubhIDs = ['{0}/{1}'.format(Chunked_loc,i) for i in ChunkedSubhIDs if 'ShIDList_Snap{0}_Chunk'.format(snap_number) in i]
        ChunkedSubhIDs.sort()

    
        
        
    #######################################################
    #######################################################
    ##Check that file to store data dictionary in exists.##
    ##If it doesn't, create it and initialise it.        ##
    ##If it does, load it.                               ##
    #######################################################
    #######################################################
    
    #outfile_name = '/u/cwalker/Illustris_Zhang_Method/Sim_{0}_Snap_{1}_dDMdz_Output.npy'.format(sim_to_use,snap_number) #name of this file
    outfile_name = '/u/cwalker/Illustris_Zhang_Method/Sim_{0}_Snap_{1}_dDMdz_Output_pID_test.npy'.format(sim_to_use,snap_number) #name of this file

     
    #####################################
    #check to see if file already exists#
    #####################################
        
    if not os.path.isfile('{0}'.format(outfile_name)):
        print('Warning: file {0} does not yet exist. Must be created.'.format(outfile_name))
        existcheck = False
    else:
        print('Note: File {0} already exists. Will be loaded.'.format(outfile_name))
        existcheck = True
        
    ######################
    #if file exists, load#
    ######################
    if existcheck == True:
        print('Loading file ({0})'.format(outfile_name))
        dict_to_edit = np.load(outfile_name,allow_pickle=True).tolist()
        print('File loaded has {0} keys'.format(len(dict_to_edit)))
    
    
    ##############################################
    #if file doesn't exist, create and initialise#
    ##############################################
    
    if existcheck == False:
        print('Creating file ({0})'.format(outfile_name))
        dict_to_edit = {} # initialise
        
        #initialise keys to be stored
        dict_to_edit['dDMdz_Zhang'] = []
        dict_to_edit['dDMdzHalo_Zhang'] = []
        dict_to_edit['dDMdzFilament_Zhang'] = []
        dict_to_edit['dDMdzVoid_Zhang'] = []
        dict_to_edit['nHalo_Zhang'] = []
        dict_to_edit['nFilament_Zhang'] = []
        dict_to_edit['nVoid_Zhang'] = []

        dict_to_edit['dDMdz_Pakmor'] = []
        dict_to_edit['dDMdzHalo_Pakmor'] = []
        dict_to_edit['dDMdzFilament_Pakmor'] = []
        dict_to_edit['dDMdzVoid_Pakmor'] = []
        dict_to_edit['nHalo_Pakmor'] = []
        dict_to_edit['nFilament_Pakmor'] = []
        dict_to_edit['nVoid_Pakmor'] = []
        
        #edit 09/02/22 for storing information about subhalos along sightline
        dict_to_edit['firstShID'] = [] #first subhalo ID number along pipe line of sight
        dict_to_edit['uniqueShIDs'] = [] #unique subhalo ID numbers along pipe line of sight
        dict_to_edit['closestCoords'] = [] #closest coordinates along pipe line of sight to these subhalos
        
        #save
        np.save('{0}'.format(outfile_name),dict_to_edit)
        print('File created and initialised')
        
        
    #######################################################
    #check to see if file contains correct number of pipes#
    #######################################################
    
    if len(dict_to_edit['dDMdz_Pakmor'])<npipes:
        print('Warning: File currently contains too few pipes ({0}/{1})'.format(len(dict_to_edit['dDMdz_Pakmor']),npipes))
        lencheck = False
    
    elif len(dict_to_edit['dDMdz_Pakmor'])==npipes:
        print('Warning: File already contains the correct number of pipes ({0}). No more will be created'.format(len(dict_to_edit['dDMdz_Pakmor'])))
        lencheck = True
    
    ###################################################################
    #if number of pipes is too low, calculate how many more are needed#
    ###################################################################
    
    if lencheck == False:
        new_npipes = npipes - len(dict_to_edit['dDMdz_Pakmor'])
        print('Remaining number of pipes needed is: {0}'.format(new_npipes))
    
    #####################################
    #if number of pipes is correct, exit#
    #####################################
    
    if lencheck == True:
        print('No new pipes needed. Quitting program.')
        #break

    
    ###########################
    ###########################
    ##create pipes, get dDMdz##
    ###########################
    ###########################
    elif lencheck==False:
        while(len(dict_to_edit['dDMdz_Pakmor'])<npipes): #while not enough pipes have been created:

            #############
            #Create Pipe#
            #############

            #HACK FOR TESTING TO MAKEE SURE IT ONLY LOOPS ONCE. REMOVE BEFORRE PUTTING IN SCRIPT!
            #npipes = 0

            #########################################
            #define los coordinates at start of pipe#
            #########################################

            #By Zhang+20 definition of following x-axis,
            #x will be zero, y and z will be random
            #units default = ckpc/h (compare box size to https://www.tng-project.org/about/)

            pipe_start_coords = np.array([0,
                                 np.random.uniform(0,header['BoxSize'],1)[0],
                                 np.random.uniform(0,header['BoxSize'],1)[0]])
            #print('Random start cell coordinates: {0}'.format(pipe_start_coords))

            ###################################
            #define coordinates at end of pipe#
            ###################################

            #By Zhang+20 definition of following x-axis,
            #x will be length of simulation,y and z will be same as start coords

            pipe_end_coords = pipe_start_coords+np.array([header['BoxSize'],0,0])
            #print('Pipe end cell coordinates: {0}'.format(pipe_end_coords))


            ########################
            #plot the line of sight#
            ########################

            los_toplot=list(zip(pipe_start_coords,pipe_end_coords))

            ########################
            #construct pipe corners#
            ########################

            #Add and subtract half of pipe length from y and z coords for y and z boundaries
            #code adapted from https://stackoverflow.com/questions/33540109/plot-surfaces-on-a-cube

            c1s = pipe_start_coords + np.array([0,pipe_width/2,pipe_width/2]) #start corner 1
            c2s = pipe_start_coords + np.array([0,-pipe_width/2,-pipe_width/2]) #start corner 2
            c3s = pipe_start_coords + np.array([0,pipe_width/2,-pipe_width/2]) #start corner 3
            c4s = pipe_start_coords + np.array([0,-pipe_width/2,pipe_width/2]) #start corner 4

            c1e = pipe_end_coords + np.array([0,pipe_width/2,pipe_width/2]) #end corner 1
            c2e = pipe_end_coords + np.array([0,-pipe_width/2,-pipe_width/2]) #end corner 2
            c3e = pipe_end_coords + np.array([0,pipe_width/2,-pipe_width/2]) #end corner 3
            c4e = pipe_end_coords + np.array([0,-pipe_width/2,pipe_width/2]) #end corner 4

            corners = np.array([c1s,c2s,c3s,c4s,c1e,c2e,c3e,c4e])

            ######################
            #construct pipe edges#
            ######################

            line1 = list(zip(c1s,c1e))
            line2 = list(zip(c2s,c2e))
            line3 = list(zip(c3s,c3e))
            line4 = list(zip(c4s,c4e))
            line5 = list(zip(c1s,c3s))
            line6 = list(zip(c3s,c2s))
            line7 = list(zip(c2s,c4s))
            line8 = list(zip(c4s,c1s))
            line9 = list(zip(c1e,c3e))
            line10 = list(zip(c3e,c2e))
            line11 = list(zip(c2e,c4e))
            line12 = list(zip(c4e,c1e))

            lines_todraw = np.array([line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11,line12])

            ###########################################
            #get cells in this pipe by partial loading#
            ###########################################

            ###########################################
            ###########################################
            ##Parallelisation edit of the code begins##
            ###########################################
            ###########################################

            
            if parallelcodetest == True:
                print('Running non parallel version of code to test functions')
                
            #test  of functions
                cpu_map_a = np.arange(n_full_core*cpus_to_use).reshape(cpus_to_use,n_full_core)
                cpu_map_b = np.arange(n_full_core*cpus_to_use,nSubLoads).reshape(n_partial_core,1)
                package_a = [(snap_number,basePath,sim_to_use,nSubLoads,cpu_map_a[i],T_h,T_c,c1s,c2e) for i in range(cpus_to_use)]
                package_b = [(snap_number,basePath,sim_to_use,nSubLoads,cpu_map_b[i],T_h,T_c,c1s,c2e) for i in range(n_partial_core)]
                print('Testing CPU maps: A: {0}\n B: {1}'.format(cpu_map_a,cpu_map_b))
                print('Testing packages: A: {0}\n B: {1}'.format(package_a,package_b))
                #running one package through the code
                print('Package to run (A): {0}'.format(package_a[0]))
                print('Package to run (B): {0}'.format(package_b[0]))
                print('\nrunning A...\n')
                for test_i in range(len(package_a)):
                    unwrap_package(package_a[test_i])
                print('\nrunning B...\n')
                for test_i in range(len(package_b)):
                    unwrap_package(package_b[test_i])
                print('\nRan successfully')
                
            elif parallelcodetest==False:
                print('Running parallelised version to check parallelisation')
            
                #create cpu_map and packages for processing
                #this array dictates which sections of the data a cpu will load

                if n_partial_core ==0: #if there are no remaining parts to load after the full core runs:
                    #cpu map
                    cpu_map = np.arange(n_full_core*cpus_to_use).reshape(cpus_to_use,n_full_core)
                    #the package to be unwrapped for multiprocessing
                    package = [(snap_number,basePath,sim_to_use,nSubLoads,cpu_map[i],T_h,T_c,c1s,c2e) for i in range(cpus_to_use)]

                    print(cpu_map)

                    with closing(Pool(cpus_to_use)) as p: #invoke multiproccessing
                        run = p.map(unwrap_package,package,chunksize=1) #run the multiprocessing
                    p.terminate() #terminate after completion

                elif n_partial_core > 0: #if there are remaining parts to load after the full core runs:
                    #cpu map for full core runs
                    cpu_map_a = np.arange(n_full_core*cpus_to_use).reshape(cpus_to_use,n_full_core)
                    #package for full core runs
                    package_a = [(snap_number,basePath,sim_to_use,nSubLoads,cpu_map_a[i],T_h,T_c,c1s,c2e) for i in range(cpus_to_use)]
                    #cpu map for partial core run
                    cpu_map_b = np.arange(n_full_core*cpus_to_use,nSubLoads).reshape(n_partial_core,1)
                    #package for full core runs
                    package_b = [(snap_number,basePath,sim_to_use,nSubLoads,cpu_map_b[i],T_h,T_c,c1s,c2e) for i in range(n_partial_core)]

                    print('a',cpu_map_a,package_a,'b',cpu_map_b,package_b)

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
            
            
            ###########################################################
            #loop over stored temporary files for each part, load data#
            ###########################################################
            
            #initialise arrays to hold all loaded data

            all_coords = [] #coordinates
            all_dens  = [] #density
            all_elab  = [] #electron abundance
            all_sfr   = [] #star formation rate
            all_dark  = [] #dark matter density
            all_warm  = [] #warm phase gas mass fraction
            all_pIDs  = [] #particle ID number
            
            for i in range(nSubLoads): #loop over parts

                #load file
                toload_filename = '/u/cwalker/Illustris_Zhang_Method/temp_chunks/sim_{0:02d}_snap_{1:03d}_cID_{2}.npy'.format(snap_number,i,sim_to_use) 
                loaded_dict = np.load(toload_filename,allow_pickle=True).tolist()

                #append data to array
                all_coords.append(loaded_dict['Coordinates'])
                all_dens.append(loaded_dict['Density'])
                all_elab.append(loaded_dict['ElectronAbundance'])
                all_sfr.append(loaded_dict['StarFormationRate'])
                all_dark.append(loaded_dict['SubfindDMdensity'])
                all_warm.append(loaded_dict['Warm'])
                all_pIDs.append(loaded_dict['ParticleIDs'])
                
                #remove temporary file after loading
                os.remove(toload_filename)

            #############################
            #flatten into correct format#
            #############################

            pipe_cell_coords = np.array([item for sublist in all_coords for item in sublist])
            pipe_cell_dens = np.array([item for sublist in all_dens for item in sublist])
            pipe_cell_elab = np.array([item for sublist in all_elab for item in sublist])
            pipe_cell_sfr = np.array([item for sublist in all_sfr for item in sublist])
            pipe_cell_dark = np.array([item for sublist in all_dark for item in sublist])
            pipe_cell_warm = np.array([item for sublist in all_warm for item in sublist])
            pipe_cell_pIDs = np.array([item for sublist in all_pIDs for item in sublist])
            
            
            ##################################
            ##################################
            ##parallelisation edit ends here##
            ##################################
            ##################################

            ############################
            ############################
            ##partial load insert ends##
            ############################
            ############################
            
            ############################
            ############################
            ##subhalo ID insert begins##
            ############################
            ############################
            
            #####################################
            #Convert particle IDs to subhalo ids#
            #####################################
            
            if pIDshID_version == 'old':
                print('old shID code')
                #run old version of the particle ID to subhalo ID conversion
                pipe_cell_shIDs = oldpIDshIDconverter(pipe_cell_pIDs,AllPartIDs,AllSubhIDs)
                
            elif pIDshID_version == 'new':
                print('new shID code')
                #run the new version of the particle ID to subhalo ID conversion
                pipe_cell_shIDs = newpIDshIDconverter(pipe_cell_pIDs,ChunkedPartIDs,ChunkedSubhIDs)
                
            elif pIDshID_version == 'both':
                print('comparing both shID codes')
                #run old version of the particle ID to subhalo ID conversion
                pipe_cell_shIDs_1 = oldpIDshIDconverter(pipe_cell_pIDs,AllPartIDs,AllSubhIDs)
                #run the new version of the particle ID to subhalo ID conversion
                pipe_cell_shIDs_2 = newpIDshIDconverter(pipe_cell_pIDs,ChunkedPartIDs,ChunkedSubhIDs)
                #compare both versions
                array_equal_test = np.array_equal(pipe_cell_shIDs_1,pipe_cell_shIDs_2)
                if array_equal_test==True:
                    print('Arrays are the same')
                else:
                    print('error: results are not the same')
                    print(pipe_cell_shIDs_1,pipe_cell_shIDs_2)
                    break
                    
            ##########################
            ##########################
            ##subhalo ID insert ends##
            ##########################
            ##########################
            
            ############################################################
            #For pure Zhang+20 method, exclude all star forming regions#
            ############################################################

            pipe_cell_coords_z = pipe_cell_coords[np.where(pipe_cell_sfr==0)]
            pipe_cell_dens_z = pipe_cell_dens[np.where(pipe_cell_sfr==0)]
            pipe_cell_elab_z = pipe_cell_elab[np.where(pipe_cell_sfr==0)]
            pipe_cell_sfr_z = pipe_cell_sfr[np.where(pipe_cell_sfr==0)]
            pipe_cell_dark_z = pipe_cell_dark[np.where(pipe_cell_sfr==0)]
            pipe_cell_pIDs_z = pipe_cell_pIDs[np.where(pipe_cell_sfr==0)]
            pipe_cell_shIDs_z = pipe_cell_shIDs[np.where(pipe_cell_sfr==0)]

            #print('sum for star forming check: {0}'.format(pipe_cell_sfr_z.sum()))

            #############################################################################################
            #For Pakmor+18 method, apply correction to A for star forming regions and leave no cells out#
            #############################################################################################

            pipe_cell_coords_p = pipe_cell_coords[:]
            pipe_cell_dens_p   = pipe_cell_dens[:]
            pipe_cell_elab_p   = pipe_cell_elab[:]*pipe_cell_warm[:] #perform Pakmor correction
            pipe_cell_sfr_p    = pipe_cell_sfr[:]
            pipe_cell_dark_p   = pipe_cell_dark[:]
            pipe_cell_pIDs_p   = pipe_cell_pIDs[:]
            pipe_cell_shIDs_p  = pipe_cell_shIDs[:]
            
            #print(pipe_cell_pIDs_p)
            #print(pipe_cell_shIDs_p)

            ###############################################
            #divide pipe into 10,000 bins along the x-axis#
            ###############################################

            #Question: why 10,000 bins given there are so few particles in the pipe?

            pipe_x_bins = np.linspace(pipe_start_coords[0],pipe_end_coords[0],nbins)
            #print('Pipe x-axis bin coordinates: {0} ckpc/h'.format(pipe_x_bins))

            #######################################
            #get coordinates of center of each bin#
            #######################################

            pipe_bin_coords = np.array([[i,pipe_start_coords[1],pipe_start_coords[2]]for i in pipe_x_bins])


            ###############################################################
            #for each bin, find distance between it and every cell in pipe#
            #find the one with miniimum distance                          #
            #this will be the cell in the los                             #
            #do for zhang (excluding sfr) and non-zhang (including sfr)   #
            ###############################################################


            ###########
            #Pakmor   #
            ###########

            #initialise empty array to hold indices of closest particle to each bin
            nearest_idxs_p = []

            for i in range(len(pipe_bin_coords)): #loop over bins
                coords = pipe_bin_coords[i] #get bin coordinates
                distarr = np.sqrt(np.sum(((pipe_cell_coords_p[:]-coords)**2),axis=1)) #create array of distances from cells
                nearest = np.argmin(distarr) #find nearest cell to bin
                nearest_idxs_p.append(nearest) #append to array

            nearest_idxs_p = np.array(nearest_idxs_p) #convert to numpy array
            nearest_idxs_unique_p = np.unique(nearest_idxs_p) #some cells are the closest to multiple bins. Get uniques.

            ##############
            #zhang method#
            ##############

            #initialise empty array to hold indices of closest particle to each bin
            nearest_idxs_z = []

            for i in range(len(pipe_bin_coords)): #loop over bins
                coords = pipe_bin_coords[i] #get bin coordinates
                distarr = np.sqrt(np.sum(((pipe_cell_coords_z[:]-coords)**2),axis=1)) #create array of distances from cells
                nearest = np.argmin(distarr) #find nearest cell to bin
                nearest_idxs_z.append(nearest) #append to array

            nearest_idxs_z = np.array(nearest_idxs_z) #convert to numpy array
            nearest_idxs_unique_z = np.unique(nearest_idxs_z) #some cells are the closest to multiple bins. Get uniques.

            #print('Nearest {0} particle ids: {1}'.format(np.shape(nearest_idxs),nearest_idxs))
            #print('Of these, {0} are unique: {1}'.format(np.shape(nearest_idxs_unique),nearest_idxs_unique))

            #################################
            #extract data from nearest cells#
            #################################

            ###########
            #Pakmor   #
            ###########

            pipe_nearest_coords_p = np.array(pipe_cell_coords_p[nearest_idxs_p]) #coordinates [ckpc/h]
            pipe_nearest_dens_p   = np.array(pipe_cell_dens_p[nearest_idxs_p])   #densities [(1e10Msun/h)/(ckpc/h)**3]
            pipe_nearest_elab_p   = np.array(pipe_cell_elab_p[nearest_idxs_p])   #electron abundance [-]
            pipe_nearest_sfr_p    = np.array(pipe_cell_sfr_p[nearest_idxs_p])    #star formation rate [Msun/yr]
            pipe_nearest_dark_p   = np.array(pipe_cell_dark_p[nearest_idxs_p])   #comoving dark matter density [(1e10Msun/h)/(ckpc/h)**3]
            pipe_nearest_pIDs_p   = np.array(pipe_cell_pIDs_p[nearest_idxs_p])   #particle ID numbers
            pipe_nearest_shIDs_p  = np.array(pipe_cell_shIDs_p[nearest_idxs_p])  #subhalo ID numbers
            
            #######
            #zhang#
            #######
            
            pipe_nearest_coords_z = np.array(pipe_cell_coords_z[nearest_idxs_z]) #coordinates [ckpc/h]
            pipe_nearest_dens_z   = np.array(pipe_cell_dens_z[nearest_idxs_z])   #densities [(1e10Msun/h)/(ckpc/h)**3]
            pipe_nearest_elab_z   = np.array(pipe_cell_elab_z[nearest_idxs_z])   #electron abundance [-]
            pipe_nearest_sfr_z    = np.array(pipe_cell_sfr_z[nearest_idxs_z])    #star formation rate [Msun/yr]
            pipe_nearest_dark_z   = np.array(pipe_cell_dark_z[nearest_idxs_z])   #comoving dark matter density [(1e10Msun/h)/(ckpc/h)**3] 
            pipe_nearest_pIDs_z   = np.array(pipe_cell_pIDs_z[nearest_idxs_z])   #particle ID numbers
            pipe_nearest_shIDs_z  = np.array(pipe_cell_shIDs_z[nearest_idxs_z])  #particle ID numbers
            
            #############################################
            #############################################
            ##subhalo central coordinates insert begins##
            #############################################
            #############################################
            
            #get first subhalo id
            first_shID = pipe_nearest_shIDs_p[0]
            
            #get unique subhalo ids in the pipe
            unique_shIDs = np.unique(pipe_nearest_shIDs_p)
            
            #get non- negative one subhalos
            unique_shIDs_notneg1 = np.where(unique_shIDs!=-1)
            unique_shIDs_notneg1 = unique_shIDs[unique_shIDs_notneg1]
            
            #get central coordinates for subhalos with non -1 subhalo IDs
            closest_coords = [] #initialise array to store
            
            for shID in unique_shIDs_notneg1:
                print('shid: {0}, snap numberr: {1}'.format(shID,snap_number))
                gas = il.snapshot.loadSubhalo(basePath, snap_number, shID, 'gas', fields=None)
                subhalo = il.groupcat.loadSingle(basePath, snap_number, subhaloID=shID)
                centralpos = subhalo['SubhaloPos']
                print('shid central pos: {0}'.format(centralpos))
                
                #get coordinates of closest approach to these subhalo IDs
                placeholder = np.copy(pipe_bin_coords[0]) #placeholder coordinates, y and z will be equal to sightline
                placeholder[0] = centralpos[0]   #set x position equal to that of subhalo center.
                print('closest: {0}'.format(placeholder))
                closest_coords.append(placeholder)
            print('unique subhalo IDs in pipe: {0}'.format(unique_shIDs))
            print('placeholder coordinates for point of closest approach: {0}'.format(closest_coords))
            
            ###########################################
            ###########################################
            ##subhalo central coordinates insert ends##
            ###########################################
            ###########################################
            
            ###############################################
            #convert density to si units using artale code#
            ###############################################

            pipe_nearest_dens_p_si = TNG_Dens2SI_astropy(pipe_nearest_dens_p)
            pipe_nearest_dens_z_si = TNG_Dens2SI_astropy(pipe_nearest_dens_z)

            ###########################################################
            #convert dark matter density to si units using artale code#
            ###########################################################

            pipe_nearest_dark_p_si = TNG_Dens2SI_astropy(pipe_nearest_dark_p)         
            pipe_nearest_dark_z_si = TNG_Dens2SI_astropy(pipe_nearest_dark_z)         

            #########################################################################
            #divide dark matter density by critical density to create the LSS tracer#
            #########################################################################

            pipe_nearest_LSStracer_p = pipe_nearest_dark_p_si/my_dens_crit
            pipe_nearest_LSStracer_z = pipe_nearest_dark_z_si/my_dens_crit
            #print('The structure tracer array is {0}'.format(pipe_nearest_LSStracer_z))       

            ##########################################
            #Create Large-Scale Structure (LSS) masks#
            ##########################################

            #non-zhang
            voi_mask_PT0_p = pipe_nearest_LSStracer_p < 0.1
            fil_mask_PT0_p = np.logical_and(pipe_nearest_LSStracer_p >= 0.1, pipe_nearest_LSStracer_p < 57)#CELESTE:CORRECTED
            hal_mask_PT0_p = pipe_nearest_LSStracer_p >= 57 

            #zhang
            voi_mask_PT0_z = pipe_nearest_LSStracer_z < 0.1
            fil_mask_PT0_z = np.logical_and(pipe_nearest_LSStracer_z >= 0.1, pipe_nearest_LSStracer_z < 57)#CELESTE:CORRECTED
            hal_mask_PT0_z = pipe_nearest_LSStracer_z >= 57        

            ##############################################################
            #Calculate the number of nearest cells of each structure type#
            ##############################################################

            num_voi_cells_z = np.shape(pipe_nearest_coords_z[voi_mask_PT0_z])[0]
            num_fil_cells_z = np.shape(pipe_nearest_coords_z[fil_mask_PT0_z])[0]
            num_hal_cells_z = np.shape(pipe_nearest_coords_z[hal_mask_PT0_z])[0]

            num_voi_cells_p = np.shape(pipe_nearest_coords_p[voi_mask_PT0_p])[0]
            num_fil_cells_p = np.shape(pipe_nearest_coords_p[fil_mask_PT0_p])[0]
            num_hal_cells_p = np.shape(pipe_nearest_coords_p[hal_mask_PT0_p])[0]

            ##########################################
            #get electron density at each of the bins#
            ##########################################

            #follow zhang+20 equation exactly as native units of TNG are
            #comoving

            #############################################################
            #Zhang: pne = (ElAb)*hmasssfrac*(Dens/protonmass)*((1+z)**3)#
            #use data which excludes SFRs                               #
            #############################################################

            #total
            pipe_nearest_pne_z = (pipe_nearest_elab_z)*hmassfrac*(pipe_nearest_dens_z_si/protonmass)*((1+header['Redshift'])**3)
            pipe_nearest_pne_p = (pipe_nearest_elab_p)*hmassfrac*(pipe_nearest_dens_p_si/protonmass)*((1+header['Redshift'])**3)
            #print('pnes are: {0}'.format(pipe_nearest_pne_z))

            #halos
            pipe_nearest_pne_z_hal = (pipe_nearest_elab_z[hal_mask_PT0_z])*hmassfrac*(pipe_nearest_dens_z_si[hal_mask_PT0_z]/protonmass)*((1+header['Redshift'])**3)
            pipe_nearest_pne_p_hal = (pipe_nearest_elab_z[hal_mask_PT0_p])*hmassfrac*(pipe_nearest_dens_p_si[hal_mask_PT0_p]/protonmass)*((1+header['Redshift'])**3)
            #print('pnes in halos are: {0}'.format(pipe_nearest_pne_z_hal))

            #filaments
            pipe_nearest_pne_z_fil = (pipe_nearest_elab_z[fil_mask_PT0_z])*hmassfrac*(pipe_nearest_dens_z_si[fil_mask_PT0_z]/protonmass)*((1+header['Redshift'])**3)
            pipe_nearest_pne_p_fil = (pipe_nearest_elab_p[fil_mask_PT0_p])*hmassfrac*(pipe_nearest_dens_p_si[fil_mask_PT0_p]/protonmass)*((1+header['Redshift'])**3)
            #print('pnes in filaments are: {0}'.format(pipe_nearest_pne_z_fil))

            #voids
            pipe_nearest_pne_z_voi = (pipe_nearest_elab_z[voi_mask_PT0_z])*hmassfrac*(pipe_nearest_dens_z_si[voi_mask_PT0_z]/protonmass)*((1+header['Redshift'])**3)
            pipe_nearest_pne_p_voi = (pipe_nearest_elab_z[voi_mask_PT0_p])*hmassfrac*(pipe_nearest_dens_p_si[voi_mask_PT0_p]/protonmass)*((1+header['Redshift'])**3)
            #print('pnes in voids are: {0}'.format(pipe_nearest_pne_z_voi))


            ######################################################################
            #Non-zhang: pne = (ElAb*Warm)*hmasssfrac*(Dens/protonmass)*((1+z)**3)#
            #use all data (sfr included) and warm mass fraction                  #
            ######################################################################    

            ##################################
            #average these electron densities#
            ##################################

            #Zhang method/Pakmor method

            #total
            pipe_average_pne_z = np.mean(pipe_nearest_pne_z)
            pipe_average_pne_p = np.mean(pipe_nearest_pne_p)
            #print('Average pne is: {0}'.format(pipe_average_pne_z))   

            #halos
            pipe_average_pne_z_hal = np.sum(pipe_nearest_pne_z_hal)/nbins
            pipe_average_pne_p_hal = np.sum(pipe_nearest_pne_p_hal)/nbins
            #print('Average pne in halos is: {0}'.format(pipe_average_pne_z_hal))

            #filaments
            pipe_average_pne_z_fil = np.sum(pipe_nearest_pne_z_fil)/nbins
            pipe_average_pne_p_fil = np.sum(pipe_nearest_pne_p_fil)/nbins
            #print('Average pne in filaments is: {0}'.format(pipe_average_pne_z_fil))

            #voids
            pipe_average_pne_z_voi = np.sum(pipe_nearest_pne_z_voi)/nbins
            pipe_average_pne_p_voi = np.sum(pipe_nearest_pne_p_voi)/nbins
            #print('Average pne in voids is: {0}'.format(pipe_average_pne_z_voi))


            ################################
            #calculate dDM/dz for this pipe#
            ################################

            #outer bit of eq 7
            outer=c.c/cosmosource.H(0)
            #print(outer)

            #E(z) according to paper eq 5
            Ez = np.sqrt((0.3089*((1+header['Redshift'])**(3)))+(0.6911))
            #print(Ez)

            #denominator of eq 7
            denominator = ((1+header['Redshift'])**(2))*Ez

            #remainder of equation 7

            #total
            edens_z = pipe_average_pne_z
            ddmdz_z = outer*edens_z/denominator
            edens_p = pipe_average_pne_p
            ddmdz_p = outer*edens_p/denominator
            #print('dDM/dz = {0}'.format(ddmdz_z.to('pc*cm**(-3)')))

            #halos
            edens_z_hal = pipe_average_pne_z_hal
            ddmdz_z_hal = outer*edens_z_hal/denominator
            edens_p_hal = pipe_average_pne_p_hal
            ddmdz_p_hal = outer*edens_p_hal/denominator

            #filaments
            edens_z_fil = pipe_average_pne_z_fil
            ddmdz_z_fil = outer*edens_z_fil/denominator
            edens_p_fil = pipe_average_pne_p_fil
            ddmdz_p_fil = outer*edens_p_fil/denominator

            #voids
            edens_z_voi = pipe_average_pne_z_voi
            ddmdz_z_voi = outer*edens_z_voi/denominator
            edens_p_voi = pipe_average_pne_p_voi
            ddmdz_p_voi = outer*edens_p_voi/denominator

            ################################
            #append new data to dictionary #
            ################################

            dict_to_edit['dDMdz_Zhang'].append(ddmdz_z.to('pc*cm**(-3)').value) #append total dDM/dz to array in [pc/cc]
            dict_to_edit['dDMdzHalo_Zhang'].append(ddmdz_z_hal.to('pc*cm**(-3)').value) #append Halo value to array in [pc/cc]
            dict_to_edit['dDMdzFilament_Zhang'].append(ddmdz_z_fil.to('pc*cm**(-3)').value) #append Filament value to array in [pc/cc]
            dict_to_edit['dDMdzVoid_Zhang'].append(ddmdz_z_voi.to('pc*cm**(-3)').value) #append Void value to array in [pc/cc]
            dict_to_edit['nHalo_Zhang'].append(num_hal_cells_z) #append number of cells in halos used to get this dDM/dz value to array
            dict_to_edit['nFilament_Zhang'].append(num_fil_cells_z) #append number of cells in filaments used to get this dDM/dz value to array
            dict_to_edit['nVoid_Zhang'].append(num_voi_cells_z) #append number of cells in voids used to get this dDM/dz value to array

            dict_to_edit['dDMdz_Pakmor'].append(ddmdz_p.to('pc*cm**(-3)').value) #append total dDM/dz to array in [pc/cc]
            dict_to_edit['dDMdzHalo_Pakmor'].append(ddmdz_p_hal.to('pc*cm**(-3)').value) #append Halo value to array in [pc/cc]
            dict_to_edit['dDMdzFilament_Pakmor'].append(ddmdz_p_fil.to('pc*cm**(-3)').value) #append Filament value to array in [pc/cc]
            dict_to_edit['dDMdzVoid_Pakmor'].append(ddmdz_p_voi.to('pc*cm**(-3)').value) #append Void value to array in [pc/cc]
            dict_to_edit['nHalo_Pakmor'].append(num_hal_cells_p) #append number of cells in halos used to get this dDM/dz value to array
            dict_to_edit['nFilament_Pakmor'].append(num_fil_cells_p) #append number of cells in filaments used to get this dDM/dz value to array
            dict_to_edit['nVoid_Pakmor'].append(num_voi_cells_p) #append number of cells in voids used to get this dDM/dz value to array

            #edit 09/02/22 for storing information about subhalos along sightline
            dict_to_edit['firstShID'].append(first_shID) #append first subhalo ID number along pipe line of sight
            dict_to_edit['uniqueShIDs'].append(unique_shIDs) #append unique subhalo ID numbers along pipe line of sight
            dict_to_edit['closestCoords'].append(closest_coords) #append closest coordinates along pipe line of sight to these subhalos
        
            #########################
            #save updated dictionary#
            #########################
            np.save('{0}'.format(outfile_name),dict_to_edit)

            ###########################
            #reload updated dictionary#
            ###########################
            dict_to_edit = np.load(outfile_name,allow_pickle=True).tolist()
            print('New length = {0}'.format(len(dict_to_edit['dDMdz_Pakmor'])))

        ###############################
        ##Once snapshot is done, print#
        ###############################

        print('Completed and stored {0}\n with {1} keys of length {2}\n'.format(outfile_name,len(dict_to_edit),len(dict_to_edit['dDMdz_Pakmor'])))



