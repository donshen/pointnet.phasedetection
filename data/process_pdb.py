import numpy as np
import MDAnalysis as mda
import os
import time
import argparse
from tqdm import tqdm


class Pdb2Pts(object):
    def __init__(self, category, trans, rot):
                 
        self.category = category
        self.trans = trans
        self.rot = rot
                 
    def replicate_box(self, increment):
        """
        Input an increment array, e.g. [-1, 0, 1] for replicating the box three times in each dimension
        """
        combs = []
        for i in increment:
            for j in increment:
                for k in increment:
                    if not i == j == k == 0:
                        combs.append([i, j, k])
        return combs

    def rand_rotation_matrix(self):
        """
        Creates a random uniform rotation matrix.
        deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
        rotation. Small deflection => small perturbation.
        """
        # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

        randnums = np.random.uniform(size=(3,))

        theta, phi, z = randnums

        theta = theta * 2.0 * np.pi  # Rotation about the pole (Z).
        phi = phi * 2.0 * np.pi  # For direction of pole deflection.
        z = z * 2.0 # For magnitude of pole deflection.

        # Compute a vector V used for distributing points over the sphere
        # via the reflection I - V Transpose(V).  This formulation of V
        # will guarantee that if x[1] and x[2] are uniformly distributed,
        # the reflected points will be uniform on the sphere.  Note that V
        # has length sqrt(2) to eliminate the 2 in the Householder matrix.

        r = np.sqrt(z)
        Vx, Vy, Vz = V = (
            np.sin(phi) * r,
            np.cos(phi) * r,
            np.sqrt(2.0 - z)
            )

        st = np.sin(theta)
        ct = np.cos(theta)

        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

        # Construct the rotation matrix  ( V Transpose(V) - I ) R.

        M = (np.outer(V, V) - np.eye(3)).dot(R)
        
        return M

    def rand_periodic_translation(self, pos, Lx, Ly, Lz):

        X, Y, Z = pos[:, 0], pos[:, 1], pos[:, 2]

        X[X > Lx] -= Lx
        Y[Y > Ly] -= Ly
        Z[Z > Lz] -= Lz
        pos = np.array([X, Y, Z]).T

        vec_trans = np.array([np.random.uniform(0, Lx * 0.5), 
                              np.random.uniform(0, Ly * 0.5), 
                              np.random.uniform(0, Lz * 0.5)])
        pos += vec_trans
        X, Y, Z = pos[:,0], pos[:,1], pos[:,2]
        X[X > Lx] -= Lx
        Y[Y > Ly] -= Ly
        Z[Z > Lz] -= Lz
        
        return np.array([X, Y, Z]).T



    #pdb file must contain 1 frame
    
    def gen_pts_from_pdb(self, pdb_path, ntrans):     
        file_list = os.listdir(pdb_path)
        pdb_files = [i for i in file_list if 'pdb' in i] 
        idx = 1
        for i in tqdm(range(1, len(pdb_files) + 1)): 
            universe = mda.Universe(pdb_path + '/' + pdb_files[i - 1])
            oxygen = universe.select_atoms('name O')
            Lx, Ly, Lz = universe.dimensions[:3]
            pos_oxygen = oxygen.positions
            
            for _ in range(ntrans):   
                # Apply a random periodic translation around a vector with random x, y, and z components that are <= period
                if self.trans:       
                    pos_oxygen = self.rand_periodic_translation(pos_oxygen, Lx, Ly, Lz)               
                
                # Replicate box so that the transformed coordinates can be wrapped into the original bounding box
                if self.rot:
                    orig_pos = pos_oxygen
                    increment = [-3, -2, -1, 0, 1, 2, 3]
                    for vec in self.replicate_box(increment):
                        pos_oxygen = np.vstack([pos_oxygen,  orig_pos + np.multiply([Lx, Ly, Lz], np.array(vec))])
                    M = self.rand_rotation_matrix()
                    pos_oxygen = np.dot(pos_oxygen, M)
                    pos_oxygen = pos_oxygen[np.logical_and(pos_oxygen[:, 0] <= Lx, pos_oxygen[:, 0] >= 0)]
                    pos_oxygen = pos_oxygen[np.logical_and(pos_oxygen[:, 1] <= Ly, pos_oxygen[:, 1] >= 0)]
                    pos_oxygen = pos_oxygen[np.logical_and(pos_oxygen[:, 2] <= Lz, pos_oxygen[:, 2] >= 0)]

                file_name = f'coord_O_{opt.category}_{idx}'
                idx += 1
                
                # Save .pts files
                pts_file = open('point_clouds/' + opt.category + '/points/' + file_name + '.pts', 'w')
                for k in range(pos_oxygen.shape[0]):
                    pts_file.write(str(pos_oxygen[k, 0]) + ' ' + str(pos_oxygen[k, 1]) + ' ' + str(pos_oxygen[k, 2]) + '\n')
                pts_file.close()
                
                # Save .seg files
                seg_file = open('point_clouds/' + opt.category + '/points_label/' + file_name + '.seg', 'w')
                for k in range(pos_oxygen.shape[0]):
                    seg_file.write('1\n')
                seg_file.close()
        return 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', type=str, required=True, help='category (phase) name: sg (single gyroid), dg (double gyroid), dd (double diamond), p (plumber\'s nightmare')
    parser.add_argument('-t', '--rand_trans', action='store_true', help='whether to apply random periodic translation')
    parser.add_argument('-r', '--rand_rot', action='store_true', help='whether to apply random periodic rotation')
    parser.add_argument('-nt',
        '--ntrans', type=int, default=1, help='number of random data augmentation (translation + rotation) for each point cloud')
    opt = parser.parse_args()
    
    if not os.path.exists('point_clouds/%s/points' % opt.category):
        os.makedirs('point_clouds/%s/points' % opt.category)
    if not os.path.exists('point_clouds/%s/points_label' % opt.category):
        os.makedirs('point_clouds/%s/points_label' % opt.category)
                  
    pdb_path = 'raw/pdb/' + opt.category

    print('Processing pdb files of %s structure...' % opt.category)   
    
    print(opt.rand_trans)
    print(opt.rand_rot)
    
    tic = time.perf_counter()
    Pdb2Pts(opt.category, opt.rand_trans, opt.rand_rot).gen_pts_from_pdb(pdb_path, opt.ntrans)
    toc = time.perf_counter()
    print('Used %d ms.' % ((toc - tic) * 1000))