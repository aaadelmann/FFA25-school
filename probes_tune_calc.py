import numpy as np
import read_dat
import pandas as pd
import os 

class ProbesTuneCalc():
    def __init__(self, num_probes, probe_filepath, co_probe_filepath):
        self.num_probes = num_probes
        self.probe_filepath = probe_filepath
        self.co_probe_filepath = co_probe_filepath

    def probes_tune_calc(self, probe_list, co_probes, naff=False):
        naff_r_tunes = []
        naff_z_tunes = []
        fft_r_tunes = []
        fft_z_tunes = []
        lost_particles =[]

        num_particles = max(probe_list[0]["id"]) +1
        print("number of particles = ", num_particles)

        co_probe_alldata_df = pd.concat(co_probes)
        co_probe_alldata_df.sort_values(by=["time (s)"], inplace=True)


        # concat all the probes together and order by time
        probe_alldata_df = pd.concat(probe_list)
        probe_alldata_df.sort_values(by=["time (s)"], inplace=True)


        #select particle from df
        co_x_data = co_probe_alldata_df['# x (m)'].to_numpy()
        co_y_data = co_probe_alldata_df['y (m)'].to_numpy()
        co_z_data = co_probe_alldata_df[' z (m)'].to_numpy()

        probe1 = probe_list[0]
        probe1.sort_values(by=["time (s)"], inplace=True)
        probe1_particle1 = probe1[probe1["id"] == 0]
        probe1_particle1 = probe1_particle1['time (s)'].to_list()
        #print(probe1['time (s)'])

        probe2 = probe_list[1]
        probe2.sort_values(by=["time (s)"], inplace=True)
        probe2_particle1 = probe2[probe2["id"] == 0]
        probe2_particle1 = probe2_particle1['time (s)'].to_list()
        #print(probe1['time (s)'])

        tof = probe1_particle1[4] - probe1_particle1[3]
        #print("time of flight = ", tof)

        cell_to_cell_time = probe2_particle1[3] - probe1_particle1[3]
        #print("cell_to_cell_time ", cell_to_cell_time)


        for i in range(num_particles): 
            # re organise data into 1 particles passes through all 16 probes        
            particle_df = probe_alldata_df[probe_alldata_df["id"] == i]

            #select particle from df
            x_data = particle_df['# x (m)'].to_numpy()
            y_data = particle_df['y (m)'].to_numpy()
            z_data = particle_df[' z (m)'].to_numpy()

            if len(x_data) >= len(co_x_data):
                radial  = np.sqrt(x_data**2 + y_data**2)[:len(co_x_data)] - np.sqrt(co_x_data**2 + co_y_data**2)
                zcoords_diff = z_data[:len(co_z_data)] - co_z_data

            elif len(x_data) <= len(co_x_data):
                radial  = np.sqrt(x_data**2 + y_data**2) - np.sqrt(co_x_data**2 + co_y_data**2)[:len(x_data)]
                zcoords_diff = z_data - co_z_data[:len(z_data)]

            elif len(x_data) == len(co_x_data):
                radial  = np.sqrt(x_data**2 + y_data**2) - np.sqrt(co_x_data**2 + co_y_data**2)
                zcoords_diff = z_data - co_z_data


            if naff:
                import PyNAFF as pnf

                try:
                    naff_r_tune = pnf.naff(radial - np.mean(radial), turns=60, nterms=1, skipTurns=0, getFullSpectrum=False)
                    naff_r_tune = naff_r_tune[0][1]
                    naff_z_tune = pnf.naff(zcoords_diff - np.mean(zcoords_diff), turns=60, nterms=1, skipTurns=0, getFullSpectrum=False)
                    naff_z_tune = naff_z_tune[0][1]
                    
                    naff_r_tunes.append(naff_r_tune)
                    naff_z_tunes.append(naff_z_tune)
                except (IndexError, ValueError) as e:
                    pass
            else:
                w = np.abs(np.fft.rfft(radial))

                freqs = np.fft.rfftfreq(len(radial), tof)

                index_max_r = np.argmax(w)
                fft_r_tune_hz = freqs[index_max_r]

                z = np.abs(np.fft.fft(zcoords_diff))
                index_max_z = np.argmax(z)
                try:
                    fft_z_tune_hz = freqs[index_max_z]
                except IndexError:
                    fft_z_tune_hz = freqs[np.argmax(z[:len(z)//2])]
                fft_r_tune = fft_r_tune_hz*tof
                fft_z_tune = fft_z_tune_hz*tof

                fft_r_tunes.append(fft_r_tune)
                fft_z_tunes.append(fft_z_tune)

        print("num_lost_particles = ", len(lost_particles))

        if naff:

            return naff_r_tunes, naff_z_tunes

        else:
            return fft_r_tunes, fft_z_tunes

    def main(self, naff=False):

        probes = []
        co_probes = []
        for i in range(1, self.num_probes+1): 
            probe_df = read_dat.read_probe(filepath=self.probe_filepath, filename='PROBE'+str(i)+'.loss')
            #drop un used column to save memory
            probe_df.drop(['px ( )', 'py ( )', 'pz ( )', 'bunchNumber ( )'], axis=1, inplace=True)

            probes.append(probe_df)

            co_probes.append(read_dat.read_probe(filepath=self.co_probe_filepath, filename='PROBE'+str(i)+'.loss'))
        
        tunex,  tunez = self.probes_tune_calc(probes, co_probes, naff=naff)

        if naff:
            print("Using NAFF")
        else:
            print("Using FFT")
        print("cell tunex, = ", tunex)
        print("cell tunez = ", tunez)
        return tunex, tunez

if __name__=="__main__":

    path = os.getcwd()
    path_to_closed_orbit_probes = os.path.join(path,"co_dir/")
    print("path to closed orbit probes = ",path_to_closed_orbit_probes)
    if os.path.exists(path_to_closed_orbit_probes):
        ProbesTuneCalc(16, path, path_to_closed_orbit_probes).main()
        
    else:
        print("path ", path_to_closed_orbit_probes, " does not exist")
        print("run opal to generate the closed orbit probes, then move the probes into a sperate directory")



