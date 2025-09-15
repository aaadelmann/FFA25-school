from read_dat import read_FieldMapRPHI, read_trackOrbit, read_FieldMap, read_probe, read_FieldMapRPHI
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

kgauss2Tesla = 1e3/1e4
    
def field_hist(FieldMap_df):
    """
    Plot histogram of the Cartesian field map
    Inputs: FieldMap_df (pandas.DataFrame) - OPAL XY field map 
    """
    min_field = FieldMap_df['Bz [kGauss]'].min()*1e3/1e4
    max_field = FieldMap_df['Bz [kGauss]'].max()*1e3/1e4
    if min_field < 0:
        abs_max_field = max(max_field, abs(min_field))
        field_norm = matplotlib.colors.Normalize(-abs_max_field, abs_max_field, False)
        cmap_ = "bwr"
    else:
        field_norm = matplotlib.colors.Normalize(0, max_field, False)
        cmap_ = "Reds"

    return plt.hist2d(FieldMap_df['x [m]'], 
                FieldMap_df['y [m]'], 
                weights=FieldMap_df['Bz [kGauss]']*1e3/1e4,
                bins=[int(len(FieldMap_df['x [m]'])/1500), int(len(FieldMap_df['y [m]'])/1500)],
                cmap=cmap_,
                norm=field_norm)

def plot_field_map_and_orbit(FieldMap_df, trackOrbit_df):
    """
    Plot the orbit on top of the XY field map
    Inputs: FieldMap_df (pandas.DataFrame)   - OPAL XY field map 
            trackOrbit_df (pandas.DataFrame) - trackOrbit.dat file
    """

    field_hist(FieldMap_df=FieldMap_df)
    plt.plot(trackOrbit_df['x [m]'], trackOrbit_df['y [m]'],linewidth=1, color='g')
    print('mean radius = ',np.mean(np.sqrt(trackOrbit_df['x [m]']**2 + trackOrbit_df['y [m]']**2)))
    plt.xlim(-6,6)
    plt.ylim(-6,6)

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.colorbar()
    plt.show()
    #plt.savefig('x-y-Bz-hist.png')
    #plt.show()
    plt.close()

def plot_field_map_and_orbit_withprobe(FieldMap_df, trackOrbit_df, probes):
    """
    Plot the orbit on top of the XY field map with the locations of probes
    Inputs: FieldMap_df (pandas.DataFrame)   - OPAL XY field map 
            trackOrbit_df (pandas.DataFrame) - trackOrbit.dat file
            probes (list)                    - list of DataFrames of the probe.loss file
    """

    xcoords =[]
    Pxcoords =[]
    ycoords =[]
    Pycoords =[]
    for df in probes:
        xcoords.append(df['# x (m)'].to_numpy()[0])
        ycoords.append(df['y (m)'].to_numpy()[0])

    field_hist(FieldMap_df=FieldMap_df)
    plt.plot(trackOrbit_df['x [m]'], trackOrbit_df['y [m]'],linewidth=1, color='g')
    plt.scatter(xcoords, ycoords)
    plt.xlim(-6,6)
    plt.ylim(-6,6)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.colorbar()
    #plt.axes().set_aspect('equal')
    #plt.savefig('x-y-Bz-hist.png')
    plt.show()
    plt.close()

def plot_r_vs_phi(FieldMapRPHI_df, trackOrbit_df):
    """
    Plot the orbit on top of the  R PHI field map
    Inputs: FieldMapRPHI_df (pandas.DataFrame) - OPAL R PHI field map 
            trackOrbit_df (pandas.DataFrame)   - trackOrbit.dat file
    """


    min_field = FieldMapRPHI_df['Bz [kGauss]'].min()*kgauss2Tesla
    max_field = FieldMapRPHI_df['Bz [kGauss]'].max()*kgauss2Tesla
    if min_field < 0:
        abs_max_field = max(max_field, abs(min_field))
        field_norm = matplotlib.colors.Normalize(-abs_max_field, abs_max_field, False)
        cmap_ = "bwr"
    else:
        field_norm = matplotlib.colors.Normalize(0, max_field, False)
        cmap_ = "Reds"

    plt.hist2d(FieldMapRPHI_df['phi [degree]'], FieldMapRPHI_df['r [mm]'],
            weights=FieldMapRPHI_df['Bz [kGauss]']*kgauss2Tesla,
            bins=[int(len(FieldMapRPHI_df['r [mm]'])/2000), int(len(FieldMapRPHI_df['phi [degree]'])/2000)],
            cmap=cmap_,
            norm=field_norm)
    plt.colorbar()
    
    plt.scatter(np.arctan2(trackOrbit_df['y [m]'],trackOrbit_df['x [m]'])*180/np.pi, (trackOrbit_df['x [m]']**2 +trackOrbit_df['y [m]']**2)**0.5, marker='.',s=1, color='g', label="orbit")   
    plt.xlabel('Azimuthal angle [deg]')
    plt.ylabel('Orbit radius [m]')
    plt.grid()
    #plt.axes().set_aspect('equal')
    #plt.savefig('r-phi-Bz-hist.png')
    plt.legend(loc='best', markerscale=20)
    plt.show()
    plt.close()

def plot_phase_space(E_kin, path_to_probs, x_co=3.9910649675642533, px_co=-0.010031648542825716):
    """
    Plot the phasespace of an OPAL simautlation:
    Inputs: E_kin (int)            - KE of particle
            path_to_probs (string) - file path to the OPAL probes files with a orbit perturbed from the closed orbit
            x_co (float)           - closed orbit x coordinates
            px_co (float)          - closed orbit px coordinates
    """
    plt.figure(1)
    plt.subplot(211)
    m_0 = 938.272 
    gamma = (E_kin/m_0) +1
    beta = (1- (1/gamma**2))**0.5
    df = read_probe(filepath=path_to_probs, filename='PROBE'+str(1)+'.loss')

    plt.scatter((df['# x (m)'].to_numpy())*1000, (df['px ( )'].to_numpy())*1000/(beta*gamma), marker='.')
    plt.xlabel("x [mm]")
    plt.ylabel("Px [mrad]")

    plt.subplot(212)
    plt.scatter(df[' z (m)'].to_numpy()*1000, (df['pz ( )'].to_numpy())*1000/(beta*gamma), marker='.')
    plt.xlabel("z [mm]")
    plt.ylabel("Pz [mrad]")
    plt.show()

    print("x_co, px_co = ", x_co, px_co)

    xcoords = df['# x (m)'].to_numpy() - x_co
    pxcoords = df['px ( )'].to_numpy()/(beta*gamma) - px_co/(beta*gamma)
    zcoords = df[' z (m)'].to_numpy()
    pzcoords = df['pz ( )'].to_numpy()/(beta*gamma)


    varianceX = np.mean(xcoords**2)
    varianceZ = np.mean(zcoords**2)

    variancePx = np.mean(pxcoords**2)
    variancePz = np.mean(pzcoords**2)

    varianceXPx = np.mean(xcoords*pxcoords)
    varianceZPz = np.mean(zcoords*pzcoords)

    rms_emitX = np.sqrt(varianceX*variancePx - varianceXPx**2)
    rms_emitZ = np.sqrt(varianceZ*variancePz - varianceZPz**2)

    # calc twiss params
    betaX = varianceX/rms_emitX
    betaZ = varianceZ/rms_emitZ

    gammaX = variancePx/rms_emitX
    gammaZ = variancePz/rms_emitZ

    alphaX = -varianceXPx/rms_emitX
    alphaZ = -varianceZPz/rms_emitZ

    print(betaX*gammaX - alphaX**2)
    print(betaZ*gammaZ - alphaZ**2)

    print('emitX = ',rms_emitX)
    print('emitZ = ',rms_emitZ)

    print('betaX = ', betaX)
    print('betaZ = ', betaZ)
    print('gammaX = ', gammaX)
    print('gammaZ = ', gammaZ)
    print('alphaX = ',alphaX)
    print('alphaZ = ',alphaZ)

def main():
    root_path = os.getcwd()
    fieldmap_path = os.path.join(root_path, "data/")
    track_filename = "DF_lattice" + '-trackOrbit.dat'
    #track_filename = "FFA_FODO_lattice" + -trackOrbit.dat

    fieldmap_filename = "FieldMapXY.dat"
    fieldmap_cyclindrical_filename =  "FieldMapRPHI.dat"

    # read the track orbit
    trackOrbit_df = read_trackOrbit(filepath=root_path
                                    , filename=track_filename)

    #read the field maps
    FieldMap_df = read_FieldMap(fieldmap_path, filename=fieldmap_filename)
    FieldMapRPHI_df = read_FieldMapRPHI(fieldmap_path, filename=fieldmap_cyclindrical_filename)


    plot_field_map_and_orbit(FieldMap_df, trackOrbit_df)
    plot_r_vs_phi(FieldMapRPHI_df, trackOrbit_df)
    plot_phase_space(3, root_path)
    probes = []
    num_cells = 16# number of probes must be the same as the number of cells
    for i in range(1,num_cells+1): 
        df = read_probe(filepath=root_path, filename='PROBE'+str(i)+'.loss')
        probes.append(df)

    plot_field_map_and_orbit_withprobe(FieldMap_df=FieldMap_df, trackOrbit_df=trackOrbit_df, probes=probes)

if __name__ == "__main__":
    main()
