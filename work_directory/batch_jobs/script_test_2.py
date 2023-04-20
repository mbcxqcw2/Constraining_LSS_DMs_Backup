#Notes
#This is a copy of ../Illustris_Zhang_Method/Pipe_Creation_Plus_LSS_6.ipynb
#to be run as a script in preparation for being turned into a batch job
#It builds upon ./script_test.py by having coommand line inputs
#imports

import illustris_python as il
import numpy as np
from numpy import random as rand
from charlie_TNG_tools import temp2u
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import binned_statistic_dd
from astropy import constants as c
#from artale_constants import *
from astropy.cosmology import Planck15 as cosmosource
import h5py
import os
import sys

#parse command line arguments#
cla = sys.argv
n_inputs = 3

if len(cla)!=n_inputs+1:
    print("Error! {0} arguments required. {1} arguments provided.".format(n_inputs,len(cla)))
    print("Exiting script.")
    sys.exit()

else:
    sim_to_use = str(cla[1])# the simulation to use, e.g. 'TNG50-4'
    pipes_per_snap = int(cla[2]) #the number of pipes to create per snapshot, e.g. 5125
    snap_to_process = int(cla[3]) #the snapshot number to process, e.g. 99

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

#pipe info for test
npipes      = 1  #number of pipes to create
snap_number = 99 #snapshot number for test


#base path to simulation
basePath = '/virgo/simulations/IllustrisTNG/{0}/output/'.format(sim_to_use)

#load header
header = il.groupcat.loadHeader(basePath,snap_number)

#fields to load for test
fields=['Density',
        'ElectronAbundance',
        'StarFormationRate',
        'InternalEnergy',
        'Coordinates',
        'Masses',
        'SubfindDMDensity']

#define constants foor warm-phase gas mass fraction calculation
T_h = 10**7  #hot phase gase temperature [Kelvin]
T_c = 10**3  #cold phase gas temperature [Kelvin]
x_h = 0.75   #Hydrogen mass fraction




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
    
    #######################################################
    #######################################################
    ##Check that file to store data dictionary in exists.##
    ##If it doesn't, create it and initialise it.        ##
    ##If it does, load it.                               ##
    #######################################################
    #######################################################
    
    outfile_name = '/u/cwalker/Illustris_Zhang_Method/Sim_{0}_Snap_{1}_dDMdz_Output.npy'.format(sim_to_use,snap_number) #name of this file
     
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

            #initialise
            pipe_cell_coords_part=[]
            pipe_cell_dens_part = []
            pipe_cell_elab_part=[]
            pipe_cell_sfr_part=[]
            pipe_cell_dark_part = []
            pipe_cell_warm_part=[]

            #########################
            #loop over partial loads#
            #########################

            for j in range(nSubLoads):

                ###########################
                #load the partial data set#
                ###########################

                data = loadSubset(basePath,snap_number, 'gas', fields,chunkNum=j, totNumChunks=nSubLoads)

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

                ########################
                #get cells in this pipe#
                ########################

                yz_pts = data['Coordinates'][:,[1,2]]
                ur = c1s[1:] #upper right of pipe start (y and z only)
                ll = c2e[1:] #lower left of pipe end (y and z only)
                inidx = np.all((ll <= yz_pts) & (yz_pts <= ur), axis=1) #indexes of cells in pipe

                ###########################
                #get data of cells in pipe#
                ###########################

                pipe_cell_coords_part.append(data['Coordinates'][inidx])       #coordinates [ckpc/h]
                pipe_cell_dens_part.append(data['Density'][inidx])           #densities [(1e10Msun/h)/(ckpc/h)**3]
                pipe_cell_elab_part.append(data['ElectronAbundance'][inidx]) #electron abundance [-]
                pipe_cell_sfr_part.append(data['StarFormationRate'][inidx]) #star formation rate [Msun/yr]
                pipe_cell_dark_part.append(data['SubfindDMDensity'][inidx])  #comoving dark matter density [(1e10Msun/h)/(ckpc/h)**3]
                pipe_cell_warm_part.append(data['Warm'][inidx])

            #############################
            #flatten into correct format#
            #############################

            pipe_cell_coords = np.array([item for sublist in pipe_cell_coords_part for item in sublist])
            pipe_cell_dens   = np.array([item for sublist in pipe_cell_dens_part for item in sublist])
            pipe_cell_elab   = np.array([item for sublist in pipe_cell_elab_part for item in sublist])
            pipe_cell_sfr    = np.array([item for sublist in pipe_cell_sfr_part for item in sublist])
            pipe_cell_dark   = np.array([item for sublist in pipe_cell_dark_part for item in sublist])
            pipe_cell_warm   = np.array([item for sublist in pipe_cell_warm_part for item in sublist])

            ############################
            ############################
            ##partial load insert ends##
            ############################
            ############################

            ############################################################
            #For pure Zhang+20 method, exclude all star forming regions#
            ############################################################

            pipe_cell_coords_z = pipe_cell_coords[np.where(pipe_cell_sfr==0)]
            pipe_cell_dens_z = pipe_cell_dens[np.where(pipe_cell_sfr==0)]
            pipe_cell_elab_z = pipe_cell_elab[np.where(pipe_cell_sfr==0)]
            pipe_cell_sfr_z = pipe_cell_sfr[np.where(pipe_cell_sfr==0)]
            pipe_cell_dark_z = pipe_cell_dark[np.where(pipe_cell_sfr==0)]

            #print('sum for star forming check: {0}'.format(pipe_cell_sfr_z.sum()))

            #############################################################################################
            #For Pakmor+18 method, apply correction to A for star forming regions and leave no cells out#
            #############################################################################################

            pipe_cell_coords_p = pipe_cell_coords[:]
            pipe_cell_dens_p   = pipe_cell_dens[:]
            pipe_cell_elab_p   = pipe_cell_elab[:]*pipe_cell_warm[:] #perform Pakmor correction
            pipe_cell_sfr_p    = pipe_cell_sfr[:]
            pipe_cell_dark_p   = pipe_cell_dark[:]

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
            pipe_nearest_dark_p   = np.array(pipe_cell_dark_p[nearest_idxs_p])   #comooving dark matter density [(1e10Msun/h)/(ckpc/h)**3]

            #######
            #zhang#
            #######
            pipe_nearest_coords_z = np.array(pipe_cell_coords_z[nearest_idxs_z]) #coordinates [ckpc/h]
            pipe_nearest_dens_z   = np.array(pipe_cell_dens_z[nearest_idxs_z])   #densities [(1e10Msun/h)/(ckpc/h)**3]
            pipe_nearest_elab_z   = np.array(pipe_cell_elab_z[nearest_idxs_z])   #electron abundance [-]
            pipe_nearest_sfr_z    = np.array(pipe_cell_sfr_z[nearest_idxs_z])    #star formation rate [Msun/yr]
            pipe_nearest_dark_z   = np.array(pipe_cell_dark_z[nearest_idxs_z])   #comooving dark matter density [(1e10Msun/h)/(ckpc/h)**3] 

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



