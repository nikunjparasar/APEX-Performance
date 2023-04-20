import numpy as np
import os

class Parse_RaceTracks:
    @staticmethod
    def parse(track_name):
        cwd = os.getcwd()
    
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Austin.csv')
        austin_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Austin.csv')
        austin_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'BrandsHatch.csv')
        bh_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'BrandsHatch.csv')
        bh_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Budapest.csv')
        budapest_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Budapest.csv')
        budapest_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Catalunya.csv')
        cat_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Catalunya.csv')
        cat_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Hockenheim.csv')
        hock_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Hockenheim.csv')
        hock_line = np.genfromtxt(line_path, delimiter=',')    
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'IMS.csv')
        ims_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'IMS.csv')
        ims_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Melbourne.csv')
        mel_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Melbourne.csv')
        mel_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'MexicoCity.csv')
        mc_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'MexicoCity.csv')
        mc_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Montreal.csv')
        montreal_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Montreal.csv')
        montreal_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Monza.csv')
        monza_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Monza.csv')
        monza_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'MoscowRaceway.csv')
        moscow_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'MoscowRaceway.csv')
        moscow_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Norisring.csv')
        noris_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Norisring.csv')
        noris_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Nuerburgring.csv')
        burg_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Nuerburgring.csv')
        burg_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Oschersleben.csv')
        osch_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Oschersleben.csv')
        osch_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Sakhir.csv')
        sakhir_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Sakhir.csv')
        sakhir_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'SaoPaulo.csv')
        sp_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'SaoPaulo.csv')
        sp_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Sepang.csv')
        sepang_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Sepang.csv')
        sepang_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Shanghai.csv')
        shanghai_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Shanghai.csv')
        shanghai_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Silverstone.csv')
        silv_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Silverstone.csv')
        silv_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Sochi.csv')
        sochi_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Sochi.csv')
        sochi_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Spa.csv')
        spa_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Spa.csv')
        spa_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Spielberg.csv')
        spiel_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Spielberg.csv')
        spiel_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Suzuka.csv')
        suzuka_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Suzuka.csv')
        suzuka_line = np.genfromtxt(line_path, delimiter=',')    
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'YasMarina.csv')
        yasm_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'YasMarina.csv')
        yasm_line = np.genfromtxt(line_path, delimiter=',')
        file_path = os.path.join(cwd, 'Processing/TrackModels', 'Zandvoort.csv')
        zandvoort_data = np.genfromtxt(file_path, delimiter=',')
        line_path = os.path.join(cwd, 'Processing/Lines', 'Zandvoort.csv')
        zandvoort_line = np.genfromtxt(line_path, delimiter=',')
        
        
        # also why tf python not have a switch case
        
        if track_name == 'track_01':
            track_data = austin_data
            line_data = austin_line
        elif track_name == 'track_02':
            track_data = bh_data
            line_data = bh_line
        elif track_name == 'track_03':
            track_data = budapest_data
            line_data = budapest_line
        elif track_name == 'track_04':
            track_data = cat_data
            line_data = cat_line
        elif track_name == 'track_05':
            track_data = hock_data
            line_data = hock_line
        elif track_name == 'track_06':
            track_data = ims_data
            line_data = ims_line
        elif track_name == 'track_07':
            track_data = mel_data    
            line_data = mel_line
        elif track_name == 'track_08':
            track_data = mc_data
            line_data = mc_line
        elif track_name == 'track_09':
            track_data = montreal_data
            line_data = montreal_line
        elif track_name == 'track_10':
            track_data = monza_data
            line_data = monza_line
        elif track_name == 'track_11':
            track_data = moscow_data
            line_data = moscow_line
        elif track_name == 'track_12':
            track_data = noris_data
            line_data = noris_line
        elif track_name == 'track_13':
            track_data = burg_data    
            line_data = burg_line
        elif track_name == 'track_14':
            track_data = osch_data
            line_data = osch_line
        elif track_name == 'track_15':
            track_data = sakhir_data
            line_data = sakhir_line
        elif track_name == 'track_16':
            track_data = sp_data
            line_data = sp_line
        elif track_name == 'track_17':
            track_data = sepang_data
            line_data = sepang_line
        elif track_name == 'track_18':
            track_data = shanghai_data
            line_data = shanghai_line
        elif track_name == 'track_19':
            track_data = silv_data    
            line_data = silv_line
        elif track_name == 'track_20':
            track_data = sochi_data
            line_data = sochi_line
        elif track_name == 'track_21':
            track_data = spa_data
            line_data = spa_line
        elif track_name == 'track_22':
            track_data = spiel_data
            line_data = spiel_line
        elif track_name == 'track_23':
            track_data = suzuka_data
            line_data = suzuka_line
        elif track_name == 'track_24':
            track_data = yasm_data   
            line_data = yasm_line
        elif track_name == 'track_25':
            track_data = zandvoort_data   
            line_data = zandvoort_line
            
        return track_data, line_data
            