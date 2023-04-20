#This library contains functions developed by Charlie for Illustris TNG applications.

import numpy as np

def pID2shID(partIDs,partType,SubhaloLenType,SnapOffsetsSubhalo,debug=False,flagFuzz=True):
    """
    Corrected version of inverseMapPartIndicesToSubhaloIDs() from:
    https://www.tng-project.org/data/forum/topic/274/match-snapshot-particles-with-their-halosubhalo/
    
    For particles of ID number partIDs, of particle type partType
    compute the subhalo ID to which each particle index belongs. 
        
    If flagFuzz is True (default), particles in FoF fuzz are marked as outside any subhalo,
    otherwise they are attributed to the closest (prior) subhalo.
    
    INPUTS:
    
    partIDs           : [arr of ints]            The particle ID numbers to calculate Subhalo IDs for.
    
    partType             : [int]                    An integer value referring to particle type to be used 
                                                  in calculations. For gas particles, use partType = 0
                                            
    SubhaloLenType     : [2D array of ints]       The total number of member particles in each Subhalo in a
                                                  given snapshot.
    
                                                  Dimension information:
                                                  
                                                  -Will be of size: [Number of subhalos, 6]
                                                  -The second index refers to the particle types (0 = gas particles, etc.)
                                                  -Therefore SubhaloLenType[:,0] will give the total number of GAS member
                                                   particles in each subhalo in the snapshot.
                                            
                                                  See https://www.tng-project.org/data/docs/specifications/ for 
                                                  information about the structure of SubhaloLenType.
                                            
                                                  SubhaloLenType for a snapshot may be acquired via:
                                            
                                                  >subhalos = il.groupcat.loadSubhalos(basePath,snapshot,fields=['SubhaloLenType'])
                                                  >SubhaloLenType = np.copy(subhalos['SubhaloLenType'])

                                            
    SnapOffsetsSubhalo : [2D array of ints]       The 'offsets' of each subhalo into a snapshot
                                                  i.e. the indexes which refer to the 'offsets' (number of cells)
                                                  into the data of a snapshot which each subhalo in the snapshot resides.
                                              
                                                  Dimension information:
                                                  -Same as SubhaloLenType
                                                  -Therefore SnapOffsetsSubhalo[:,0] will give the number of GAS cells into
                                                   the data of a snapshot which each subhalo resides
                                                 
                                                  See https://www.tng-project.org/data/docs/specifications/#sec3a
                                                  for more information about offsets files.
                                                  
                                                  SnapOffsetsSubhalo for a snapshot may be acquired via:
                                                  
                                                  >offsetFile='/virgo/simulations/IllustrisTNG/TNG100-3/postprocessing/offsets/offsets_0{0}.hdf5'.format(snapshot_numbr)
                                                  >with h5py.File(offsetFile,'r') as f:
                                                  >    SnapOffsetsSubhalo= np.copy(f['/Subhalo/SnapByType'])
                                                  
                                                     
    debug              : [Boolean (default=False)] Charlie note: must understand what debug does.
    
    flagfuzz           : [Boolean (default=True)]  If flagFuzz is True (default), particles in FoF fuzz 
                                                   are marked as outside any subhalo,
                                                   otherwise they are attributed to the closest (prior) subhalo.
                                                  
                                                   Charlie note: must understand the FoF fuzz
    
    
    
    OUTPUTS:
    
    
    shIDs : [array of ints] the subhalo ID for each partID
    """
    

    #get number of (CHOSEN TYPE) member particles in all subhalos
    SubhaloSizes = SubhaloLenType[:,partType] 
    
    #get (CHOSEN TYPE) cell offsets into data for each subhalo
    SubhaloOffsets = SnapOffsetsSubhalo[:,partType]
    
    
    
    
    
    
    # calculate shIDs
    
    # Notes:
    # A) shIDs contains the indices of SubhaloOffsets such that, if each partIDs was inserted
    # into SubhaloOffsets just -before- its index, the order of SubhaloOffsets is unchanged
    
    # B) code does (SubhaloOffsets-1) so that the case of the particle index equaling the
    # subhalo offset (i.e. first particle) works correctly
    
    # C) code does np.ss()-1 to shift to the previous subhalo, since we want to know the
    # subhalo offset index -after- which the particle should be inserted
    
    shIDs = np.searchsorted( SubhaloOffsets - 1, partIDs ) - 1
    shIDs = shIDs.astype('int32')

    
    #case: flagFuzz = True
    
    # search and flag all matches where the indices exceed the length of the
    # subhalo they have been assigned to, e.g. either in fof fuzz, in subhalos with
    # no particles of this type, or not in any subhalo at the end of the file
    
    # In any of these cases, replace the subhalo index with -1
    
    if flagFuzz:
        
        
        #given thee definitions of SubhaloSizes and gcOffstsType,
        #gcOffsetsMax is therefore the second-to-last gas cell index into the snapshot 
        #belonging to a given subhalo (start+length-1). 
        gcOffsetsMax = SubhaloOffsets + SubhaloSizes - 1
        
        
        ww = np.where( partIDs > gcOffsetsMax[shIDs] )[0]

        if len(ww):
            shIDs[ww] = -1
            
            
    #case: debug        

    if debug:
        # for all inds we identified in subhalos, verify parents directly
        for i in range(len(partIDs)):
            if shIDs[i] < 0:
                continue
            assert partIDs[i] >= SubhaloOffsets[shIDs[i]]
            if flagFuzz:
                assert partIDs[i] < SubhaloOffsets[shIDs[i]]+SubhaloSizes[shIDs[i]]
                assert SubhaloSizes[shIDs[i]] != 0
    
    return shIDs

def temp2u(T,x_e_arr):
    """
    Converts the temperature of gas to thermal energy per unit mass
    Based on https://www.tng-project.org/data/docs/faq/
    Returned array is only valid for star forming gas
    
    Inputs:
    
    T   : (float) Temperature [Kelvin]
    x_e_arr : (array) ElectronAbundance values returned by Illustris
    
    Returns:
    
    u_arr : (array) thermal energy per unit mass 
    
    """
    
    ad_idx = 5./3             #adiabatic index
    k_B    = 1.3807 * (1e-16) #Boltzman constant in CGS units
    X_H    = 0.76             #Hydrogen mass fraction
    UEUM   = 1e10             #ratio of Illustris (UnitEnergy/UnitMass)
    m_p    = 1.6726 * (1e-24) #proton mass (cgs units?)
    
    mu_arr = (4. / (1 + (3*X_H) + (4*X_H*x_e_arr))) * m_p
    
    
    u_arr = T / ((ad_idx-1) * (1./k_B) * UEUM * mu_arr)
    
    return u_arr



############################################
#functions for multiple lightray generation#
############################################

def gen_lightray_unwrap_batch(inputdata):
    """
    A helper function for generating light rays using multiprocessing.
    Unpacks a set of data necessary for generating a batch of light rays,
    then generates them using gen_lightray_batch().
    
    INPUTS:
    
    inputdata : a list containing X arguments in the following order:
    
        ds           : yt dataset to generate light ray for
        
        lr           : light ray object which has been generated previously
        
        lr_ids       : [list of int] batch of unique identification numbers
                       for the light rays to be generated
                       
        start_coords : [list of unyt_arrays] list of unyt arrays containing
                       XYZ start coordinates for each light ray to be generated. 
                       NOTE: must be same length as lr_id
                       
        end_coords   : [list of unyt_arrays] list of unyt arrays containing
                       XYZ end coordinates for each light ray to be generated. 
                       NOTE: must be same length as lr_id
                       
        fileloc      : [str] location where light ray should be stored

        lr_tostore   : [array of str] array containing which fields should be
                       stored when creating light rays

        
    RETURNS:
    
    output of gen_lightray()
    
    """

    #unwrap the data to make the light rays

    ds            = inputdata[0]
    print ('testing unwrap batch: ds is {0}'.format(ds))
    lr            = inputdata[1]
    print ('testing unwrap batch: lr is {0}'.format(lr))
    lr_ids        = inputdata[2]
    print ('testing unwrap batch: lr_id numbers are {0}'.format(lr_ids))
    start_coords  = inputdata[3]
    print ('testing unwrap batch: start_coords are {0}'.format(start_coords))
    end_coords    = inputdata[4]
    print ('testing unwrap batch: end_coords are {0}'.format(end_coords))
    fileloc       = inputdata[5]
    print ('testing unwrap batch: output file at {0}'.format(fileloc))
    lr_tostore    = inputdata[6]
    print ('testing unwrap batch: light ray fields to store are: {0}'.format(lr_tostore))

    #add custom fields to the data set so DM can be calculated

    ds.add_field(("PartType0","proper_ne"),
             sampling_type='particle',
             function=_proper_ne,
             units="kpc**-3",
             particle_type=True,
            force_override=True)

    ds.add_field(("PartType0","x_fraction"),
            sampling_type='particle',
            function=_x_frac,
            units='dimensionless',
            particle_type=True,
            force_override=True)

    ds.add_field(("PartType0","w_fraction"),
            sampling_type='particle',
            function=_w_frac,
            units='dimensionless',
            particle_type=True,
            force_override=True)

    return gen_lightray_batch(lr,lr_ids,start_coords,end_coords,fileloc,lr_tostore)


def gen_lightray_batch(lr,lr_ids,start_coords,end_coords,fileloc,lr_tostore):
    """
    Generates a batch of light rays on a single core. Is fed from gen_lightrat_unwrap_batch()
    
    INPUTS:
                
        lr           : light ray object which has been generated previously
        
        lr_ids       : [list of int] batch of unique identification numbers
                       for the light rays to be generated
                       
        start_coords : [list of unyt_arrays] list of unyt arrays containing
                       XYZ start coordinates for each light ray to be generated. 
                       NOTE: must be same length as lr_id
                       
        end_coords   : [list of unyt_arrays] list of unyt arrays containing
                       XYZ end coordinates for each light ray to be generated. 
                       NOTE: must be same length as lr_id
                       
        fileloc      : [str] location where light ray should be stored
        
        lr_tostore   : [array of str] array containing which fields should be
                       stored when creating light rays
        
    RETURNS:
    
    output of gen_lightray_batch()
    
    """

    lr=lr

    print ('testing gen_lightray_batch: lr is {0}'.format(lr))

    #loop over light rays
    for i in range(len(lr_ids)):

        #id of the light ray
        lr_id = lr_ids[i]

        #start coordinates for light ray
        mystart=start_coords[i]
        print('testing gen_lightray_batch: lr {0} start is {1}'.format(lr_id,mystart))

        #end coordinates of the light ray
        myend=end_coords[i]
        print('testing gen_lightray_batch: lr {0} end is {1}'.format(lr_id,myend))

        #generate the light ray with appropriate fields
        ray = lr.make_light_ray(start_position=mystart,
                     end_position=myend,
                      solution_filename='{0}/lightray_{1}_solution.txt'.format(fileloc,lr_id),
                     data_filename='{0}/lightray_{1}.h5'.format(fileloc,lr_id),
                     fields=lr_tostore)

        print('testing gen_lightray_batch: lr. {0} ray is {1}'.format(lr_id,ray))

    return
