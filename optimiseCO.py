from pandas import DataFrame
from scipy.optimize import minimize
import os 
import read_dat
import numpy as np
import time

class ClosedOrbitFinder():
    def __init__(self, num_cells, input_filepath, input_filename, distribution_file):
        self.num_cells = num_cells
        self.input_filepath = input_filepath
        self.input_filename = input_filename
        self.distribution_file = distribution_file
        self.probes = None

    def run_opal(self, xPxcoords , z=0.0,Pz=0.0):
        '''
        run opal input file and then find CO by optimizing the parameters in xPxcoords, more paramters 
        may be added. This function minimizes the difference in x Px and y Py at the start and end of each cell in one turn
        ie minimizes the change in radius and radiual momentum from the start of a cell to the end.

        ''' 

        with open(self.distribution_file, 'w') as f:

            f.write('1\n') # 1 particle
            f.write(str(xPxcoords[0])+ ' '+str(xPxcoords[1])+' '+ str(0.0)+ ' '+str(0.0)+' '+ str(z)+' '+ str(Pz))
            f.close()

        os.system('/home/carl/OPAL/build/src/opal --info 0 ' + os.path.join(self.input_filepath , self.input_filename))

        #print("input command")
        #print('opal --info 0 ' + os.path.join(self.input_filepath , self.input_filename),"\n")

        # read probes
        self.probes = []
        
        # number of probes must be the same as the number of cells
        for i in range(1, self.num_cells+1):
            try:
                self.probes.append(read_dat.read_probe(self.input_filepath, filename='PROBE'+str(i)+'.loss'))
                os.remove(os.path.join(self.input_filepath,'PROBE'+str(i)+'.loss'))

            # if the 
            except FileNotFoundError:
                self.probes.append(DataFrame(data={'# x (m)':[np.random.rand()*10000], 'y (m)':[np.random.rand()*10000], 'px ( )':[np.random.rand()*10000], 'py ( )':[np.random.rand()*10000]}))

        xcoords =[]
        Pxcoords =[]
        ycoords =[]
        Pycoords =[]
        for df in self.probes:
            xcoords.append(df['# x (m)'].to_numpy()[0])
            ycoords.append(df['y (m)'].to_numpy()[0])
            Pxcoords.append(df['px ( )'].to_numpy()[0])
            Pycoords.append(df['py ( )'].to_numpy()[0])
        xcoords =np.array(xcoords)
        ycoords =np.array(ycoords)
        Pycoords =np.array(Pycoords)
        Pxcoords =np.array(Pxcoords)

        Pxcoords_rot =[]
        Pycoords_rot =[]
        phi = np.arctan2(ycoords, xcoords) +np.pi

        for k,n in enumerate(phi):
            Pxcoords_rot.append(np.cos(n)*Pxcoords[k] - np.sin(n)*Pycoords[k])
            # Px_arr is changing here so need a new variable
            Pycoords_rot.append(np.sin(n)*Pxcoords[k] + np.cos(n)*Pycoords[k])

        Pxcoords_rot =np.array(Pxcoords_rot)
        Pycoords_rot =np.array(Pycoords_rot)


        rcoords = xcoords*xcoords + ycoords*ycoords
        diff=0
        for i in range(len(rcoords)):
            diffr = (rcoords[i] - rcoords[0])**2
            diffPx = (Pxcoords_rot[i] - Pxcoords_rot[0])**2
            diffPy = (Pycoords_rot[i] - Pycoords_rot[0])**2
        
            diff += np.sqrt(diffr + diffPx + diffPy)


        print('Optimiser input = ',xPxcoords, ' | cell to cell position change = ', diffr, '| Optimisation value = ', diff,end='\r')

        return diff

    def main(self, initial_x, initial_px):
        t1 = time.time()
        res = minimize(self.run_opal, x0=[initial_x, initial_px], method='Nelder-Mead', options={'disp':True, 'xatol':1e-6, 'fatol':1e-6})

        t2 = time.time()
        print('time taken = ',t2-t1)
        
        return self.probes, res


if __name__ == "__main__":
    # example inputs
    print("THINGS TO CHECK:")
    print("Make sure this script is in the same directory as the OPAL input file and you have sourced OPAL")
    print("Make sure you've set the distribution filename correctly in the opal input file.")
    print("Make sure you've set the number of turns to 1 when finding a closed orbit")
    print("Make sure that the DUMPFIELD is set to false and the RF is off \n\n")

    # input dist file for Co at 3MeV
    distribution_filename = "CO_coords_3MeV.dat"
    with open(os.path.join(os.getcwd(), distribution_filename), 'w') as f:
        f.write('1\n') # 1 particle
        f.write(str(4.0)+ ' '+str(0.0)+' '+ str(0.0)+ ' '+str(0.0)+' '+ str(0.0)+' '+ str(0.0))
        f.close()
        
    path = os.getcwd()
    input_name = "DF_lattice"
    print("4.0 is good for 3MeV, 4.4 for 12 MeV.")
    print("Enter initial x coordinate in meters: ")
    #initial_x_guess  = 4.0
    initial_x_guess = input()
    initial_px_guess = 0.0
    ClosedOrbitFinder(16, path, input_name, distribution_filename).main(initial_x_guess, initial_px_guess)

