import numpy as np
import pandas as pd
import os
import time
import argparse
from tqdm import tqdm

class NetPts(object):
    def __init__(self, category, num_samples, trans, rot):
        self.category = category
        self.num_samples = num_samples
        self.trans = trans
        self.rot = rot
        self.dim = np.linspace(0, 1, np.random.randint(20, 30))
        self.net_param = {'dd':{'period': 2 * np.pi,
                                't': 0.6 + 0.3 * np.random.rand(), # 0.6 <= t <= 0.9
                                'fp': self.dd_pos,
                                'fn': self.dd_neg
                               },
                         'dg':{'period': 2 * np.pi,
                               't': 0.9 + 0.3 * np.random.rand(), # 0.9 <= t <= 1.2
                               'fp': self.dg_pos,
                               'fn': self.dg_neg
                              },
                         'p':{'period': 4 * np.pi,
                              't': 0.4 * np.random.rand(), # 0 <= t <= 0.4
                              'fp': self.p_pos,
                              'fn': self.p_neg
                             },
                         'sg':{'period': 2 * np.pi,
                               't': 0.6 + 0.4 * np.random.rand(), # 0.6 <= t <= 1.0
                               'fp': self.sg_pos,
                               'fn': self.sg_neg
                              },
                          }
        # minimum and maximum standard deviation of the mean-0 white noise
        self.sigma_min = .05
        self.sigma_max = .30 
        
    def rand_periodic_translation(self, pos, period):
        vec_trans = np.array([np.random.uniform(0, period * 0.5), 
                              np.random.uniform(0, period * 0.5), 
                              np.random.uniform(0, period * 0.5)])
        pos += vec_trans
        X, Y, Z = pos[:,0], pos[:,1], pos[:,2]
        X[X > period] -= period
        Y[Y > period] -= period
        Z[Z > period] -= period
        return np.array([X, Y, Z]).T


    def rand_fluctuation(self, pos, sigma_min, sigma_max):
        sigma = sigma_min + (sigma_max - sigma_min) * np.random.rand()
        X, Y, Z = pos[:, 0], pos[:, 1], pos[:, 2]
        X += np.array([np.random.uniform(-1, 1) for _ in range(len(X))]) * sigma
        Y += np.array([np.random.uniform(-1, 1) for _ in range(len(Y))]) * sigma
        Z += np.array([np.random.uniform(-1, 1) for _ in range(len(Z))]) * sigma
        return np.array([X, Y, Z]).T
    
    def replicate_box(self, increment):
        '''
        Input an increment array, e.g. [-1, 0, 1] for replicating the box three times in each dimension
        '''
        combs = []
        for i in increment:
            for j in increment:
                for k in increment:
                    if not i == j == k == 0:
                        combs.append([i, j, k])
        return combs

    def rand_rotation_matrix(self):
        """
        Creates a random rotation matrix.
        deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
        rotation. Small deflection => small perturbation.
        """
        # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

        randnums = np.random.uniform(size=(3,))

        theta, phi, z = randnums

        theta = theta * 2.0*np.pi  # Rotation about the pole (Z).
        phi = phi * 2.0*np.pi  # For direction of pole deflection.
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
    
    def dg_pos(self, x, y, z, t):
        val = np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x) - t
        return val
    
    def dg_neg(self, x, y, z, t):
        val = np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x) + t
        return val

    def sg_pos(self, x, y, z, t):
        val = np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x) - t
        return val
    
    def sg_neg(self, x, y, z, t):
        val = np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x) + t
        return val

    def dd_pos(self, x, y, z, t):
        val = (np.sin(x)*np.sin(y)*np.sin(z) + np.sin(x)*np.cos(y)*np.cos(z) + np.cos(x)*np.sin(y)*np.cos(z) +
               np.cos(x)*np.cos(y)*np.sin(z)) - t
        return val
    
    def dd_neg(self, x, y, z, t):
        val = (np.sin(x)*np.sin(y)*np.sin(z) + np.sin(x)*np.cos(y)*np.cos(z) + np.cos(x)*np.sin(y)*np.cos(z) +
               np.cos(x)*np.cos(y)*np.sin(z)) + t
        return val

    def p_pos(self, x, y, z, t):
        val = (np.cos(x) + np.cos(y) + np.cos(z)) - t
        return val 
    
    def p_neg(self, x, y, z, t):
        val = (np.cos(x) + np.cos(y) + np.cos(z)) + t
        return val
    
    def gen_net_pts(self):
        for i in tqdm(range(1, self.num_samples + 1)):
            period = self.net_param[self.category]['period']
            t = self.net_param[self.category]['t']
            x = y = z = period * self.dim
            X,Y,Z = np.meshgrid(x,y,z)
            V1 = self.net_param[self.category]['fp'](X, Y, Z, t)
            V2 = self.net_param[self.category]['fn'](X, Y, Z, t)
            region_pos1 = V1 > 0
            region_pos2 = V1 < 0
            region_neg1 = V2 > 0
            region_neg2 = V2 < 0

            if self.category == "dd" or self.category == "dg":
                region = np.logical_or(region_pos1,region_neg2)
            elif self.category == "sg":
                region = region_pos1
            elif self.category == "p":
                region = region_neg2
            else:
                print('Invalid structure name entered. Please try again.')
                break
            X_minor, Y_minor, Z_minor = X[region], Y[region], Z[region] 
            pos = np.array([X_minor,Y_minor,Z_minor]).T

            if self.trans:
                # Apply a random periodic translation around a vector with random x, y, and z components that are <= period
                pos = self.rand_periodic_translation(pos, period)
            
            if self.rot:
                # Replicate box so that the transformed coordinates can be wrapped into the original bounding box
                orig_pos = pos
                increment = [-2, -1, 0, 1, 2]
                for vec in self.replicate_box(increment):
                    pos = np.vstack([pos,  orig_pos + period * np.array(vec)])

                # Apply a random uniform rotation to the point cloud
                pos = np.dot(pos, self.rand_rotation_matrix())
                pos = pos[np.logical_and(pos[:,0] <= period, pos[:,0] >=0)]
                pos = pos[np.logical_and(pos[:,1] <= period, pos[:,1] >=0)]
                pos = pos[np.logical_and(pos[:,2] <= period, pos[:,2] >=0)]

                # Apply a random noise in the range of [0.05, 0.3] to all the points 
                pos = self.rand_fluctuation(pos, self.sigma_min, self.sigma_max)

            # Apply a random uniform rotation to the point cloud
            pts = pd.DataFrame(np.dot(pos, self.rand_rotation_matrix()))
            
            # Save .pts file
            file_name = ('coord_O_%s_%d' %(self.category,i))
            pts_file = 'point_clouds/' + opt.category + '/points/' + file_name + '.pts'
            np.savetxt(pts_file, pts.values, fmt='%.3f', delimiter=" ")
            
            # Save .seg file
            seg_file = open('point_clouds/' + opt.category + '/points_label/' + file_name + '.seg', 'w')
            for k in range(X_minor.shape[0]):
                seg_file.write('1\n')
            seg_file.close()
        return 
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
        '--category', type=str, required = True, help='category (phase) name: sg (single gyroid), dg (double gyroid), dd (double diamond), p (plumber\'s nightmare')
    parser.add_argument('-n', '--num_samples', required = True, type=int, help='how many samples to generate')
    parser.add_argument('-t', '--rand_trans', default = True, type=bool, help='whether to apply random periodic translation')
    parser.add_argument('-r', '--rand_rot', default = True, type=int, help='whether to apply random periodic rotation')
    opt = parser.parse_args()
    
    print('Generating point clouds of %s structure...' % opt.category)
    tic = time.perf_counter()
    
    if not os.path.exists('point_clouds/%s/points' % opt.category):
        os.makedirs('point_clouds/%s/points' % opt.category)
    if not os.path.exists('point_clouds/%s/points_label' % opt.category):
        os.makedirs('point_clouds/%s/points_label' % opt.category)
        
    NetPts(opt.category, opt.num_samples, opt.rand_trans, opt.rand_rot).gen_net_pts()
    toc = time.perf_counter()
    print('Used %d s.' %(toc - tic))
