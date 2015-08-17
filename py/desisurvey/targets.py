from astropy.io import fits
import numpy as np
import desimodel.focalplane

class TargetTile(object):
    """
    Keeps the relevant information for targets on a tile.

    Attributes:
         The properties initialized in the __init__ procedure:
         ra (float): array for the target's RA
         dec (float): array for the target's dec
         type (string): array for the type of target
         id (int): array of unique IDs for each target
         tile_ra (float): RA identifying the tile's center
         tile_dec (float) : dec identifying the tile's center
         tile_id (int): ID identifying the tile's ID
         n_target (int): number of targets stored in the object
         filename (string): original filename from which the info was loaded
         x (float): array of positions on the focal plane, in mm
         y (float): array of positions on the focal plane, in mm
         fiber_id (int): array of fiber_id to which the target is assigned
    """
    def __init__(self, filename):
        """
        Args:
            filename_list (str): list of strings with filenames. These filenames are expected to be in the
            FITS format for targets.
        """
    
        
        hdulist = fits.open(filename)        
        self.filename = filename
        self.ra = hdulist[1].data['RA']
        self.dec = hdulist[1].data['DEC']
        self.type = hdulist[1].data['OBJTYPE']
        self.id = np.int_(hdulist[1].data['TARGETID'])
        self.tile_ra = hdulist[1].header['TILE_RA']
        self.tile_dec = hdulist[1].header['TILE_DEC']
        self.tile_id = hdulist[1].header['TILE_ID']
        self.n = np.size(self.ra)
        fc = desimodel.focalplane.FocalPlane(ra=self.tile_ra, dec=self.tile_dec)
        self.x, self.y = fc.radec2xy(self.ra, self.dec)

        # this is related to the fiber assignment 
        self.fiber = -1.0 * np.ones(self.n, dtype='i4')

        # This section is related to the number of times a galaxy has been observed,
        # the assigned redshift and the assigned type
        self.n_observed = np.zeros(self.n, dtype='i4')
        self.assigned_z = -1.0 * np.ones(self.n)
        self.assigned_type =  np.chararray(self.n, itemsize=8)
        self.assigned_type[:] = 'NONE'

    def set_fiber(self, target_id, fiber_id):
        """
        Sets the field .fiber[] (in the target_id  location) to fiber_uid
        Args: 
            target_id (int): the target_id expected to be in self.id to modify 
                 its corresponding .fiber[] field
            fiber_id (int): the fiber_id to be stored for the corresponding target_id
        """
        loc = np.where(self.id==target_id)
        if(np.size(loc)!=0):
            loc = loc[0]
            self.fiber[loc]  = fiber_id
        else:
            raise ValueError('The fiber with %d ID does not seem to exist'%(fibers_id))

    def reset_fiber(self, target_id):
        """
        Resets the field .fiber[] (in the target_id  location) to fiber_uid
        Args: 
            target_id (int): the target_id expected to be in self.id to modify 
                 its corresponding .fiber[] field
        """
        loc = np.where(self.id==target_id)
        if(np.size(loc)!=0):
            loc = loc[0]
            self.fiber[loc]  = -1
        else:
            raise ValueError('The fiber with %d ID does not seem to exist'%(fibers_id))


    def reset_all_fibers(self):
        """
        Resets the field .fiber[] for all fibers.
        """
        self.fiber = -1.0 * np.ones(self.n, dtype='i4')


    def write_results_to_file(self, targets_file):
        """
        Writes the section associated with the results to a fits file
        Args:
            targets_file (string): the name of the corresponding targets file
        """
        
        results_file = targets_file.replace("Targets_Tile", "Results_Tile")
        if(os.path.isfile(results_file)):
            os.remove(results_file)

        c0=fits.Column(name='TARGETID', format='K', array=self.id)
        c1=fits.Column(name='NOBS', format='I', array=self.n_observed)
        c2=fits.Column(name='ASSIGNEDTYPE', format='8A', array=self.assigned_type)
        c3=fits.Column(name='ASSIGNEDZ', format='D', array=self.assigned_z)

        cat=fits.ColDefs([c0,c1,c2,c3])
        table_targetcat_hdu=fits.TableHDU.from_columns(cat)

        table_targetcat_hdu.header['TILE_ID'] = self.tile_id
        table_targetcat_hdu.header['TILE_RA'] = self.tile_ra
        table_targetcat_hdu.header['TILE_DEC'] = self.tile_dec

        hdu=fits.PrimaryHDU()
        hdulist=fits.HDUList([hdu])
        hdulist.append(table_targetcat_hdu)
        hdulist.verify()
        hdulist.writeto(results_file)

    def load_results(self, targets_file):
        """
        Loads results from the FITS file to update the arrays n_observed, assigned_z
        and assigned_type.
        Args:
            tile_file (string): filename with the target information
        """
        results_file = targets_file.replace("Targets_Tile", "Results_Tile")
        try:
            fin = fits.open(results_file)
            self.n_observed = fin[1].data['NOBS']
            self.assigned_z = fin[1].data['ASSIGNEDZ']
            self.assigned_type =  fin[1].data['ASSIGNEDTYPE']
        except Exception, e:
            import traceback
            print 'ERROR in get_tiles'
            traceback.print_exc()
            raise e

    def update_results(self, fibers):
        """
        Updates the results of each target in the tile given the 
        corresponding association with fibers.
        
        Args:
            fibers (object class FocalPlaneFibers): only updates the results if a target 
                is assigned to a fiber.
        Note:
            Right now this procedure only opdates by one the number of observations.
            It should also updated the redshift and the assigned type (given some additional information!)
        """
        for i in range(fibers.n_fiber):
            t = fibers.target[i]
            if(t != -1):
                if((t in self.id)):
                    index = np.where(t in self.id)                    
                    index = index[0]
                    self.n_observed[index]  =  self.n_observed[index] + 1
                    # these two have to be updated as well TOWRITE
                    # self.assigned_z[index] 
                    # self.assigned_type[index]                     
                else:
                    raise ValueError('The target associated with fiber_id %d does not exist'%(fibers.id[i]))


class TargetSurvey(object):
    """
    Keeps basic information for all the targets in the whole survey.

    Attributes: 
        type (string): array describing the type of target.
        id (int): 1D array of unique IDs.
        n_observed (int)
        assigned_type (string): array describing the assigned type
        assigned_z (float): array describing the redshift this source has been assigned.
        tile_names (string): list of tile's filenames where this target is present.
        ra (float): array with RA coordinates
        dec (float): array with DEC coordinates
    Note:
        This class is created for bookkeeping on the whole survey.
    """
    def __init__(self, filename_list):
        """
        Args:
            filename_list (str): list of strings with filenames. These filenames are expected to be in the
                 FITS format for targets.

        Note:
            The initialization takes a list of filenames because we expect the targets to be split by tiles.
        """
        n_file = np.size(filename_list)
        for i_file in np.arange(n_file):
            print('Adding %s to build TargetSurvey %d files to go'%(filename_list[i_file], n_file - i_file))
            tmp = TargetTile(filename_list[i_file])
            # We use the first file to initialize
            if(i_file==0):
                self.type = tmp.type.copy()
                self.id = tmp.id.copy()
                self.ra = tmp.ra.copy()
                self.dec = tmp.dec.copy()
                self.n_observed = tmp.n_observed.copy()
                self.assigned_type = tmp.assigned_type.copy()
                self.assigned_z = tmp.assigned_z.copy()
                self.tile_names= []
                for i in np.arange(np.size(self.id)):
                    self.tile_names.append([filename_list[i_file]])
            else: # the other files have to take into account the possible overlap between tiles.
                mask = np.in1d(self.id, tmp.id)

                if((len(self.tile_names)!=np.size(self.id))):
                    raise ValueError('Building TargetSurvey the numer of items in the filenames is not the same as in the ids.')
                for i in np.arange(np.size(self.id)):
                    if(mask[i]==True):
                        self.tile_names[i].append(filename_list[i_file])

                mask = ~np.in1d(tmp.id, self.id)
                n_new = np.size(np.where(mask==True))
                self.id = np.append(self.id, tmp.id[mask])
                self.type = np.append(self.type, tmp.type[mask])
                self.ra = np.append(self.ra, tmp.ra[mask])
                self.dec = np.append(self.dec, tmp.dec[mask])
                self.n_observed = np.append(self.n_observed, tmp.n_observed[mask])
                self.assigned_type = np.append(self.assigned_type, tmp.assigned_type[mask])
                self.assigned_z = np.append(self.assigned_z, tmp.assigned_z[mask])
                for i in np.arange(n_new):
                    self.tile_names.append([filename_list[i_file]])

        self.n_targets = np.size(self.id)
