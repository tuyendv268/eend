import os
from glob import glob
import multiprocessing
from tqdm import tqdm

from eend.utils.audio import Audio


class Data():
    def __init__(self, path, sample_rate):
        self.path = path
        self.sample_rate = sample_rate
        
        rttm_path = os.path.join(path, "*.rttm")
        audio_path = os.path.join("/home/tuyendv/mix_dataset/new_vivos/mix/audio", "*/audio/*.wav")
        
        # rttm_path = os.path.join(path, "rttm/*.rttm")
        # audio_path = os.path.join(path, "wav/*/audio/*.wav")
        
        self.rttm_files = glob(rttm_path)
        self.wav_files = glob(audio_path)
        self.audios = {}
        
        self.load_data_parallel()
    def gen_params(self):
        num_core = 60
        print(f"num_core: {num_core}")
        step = int(len(self.rttm_files)/num_core) + 1
        
        params = [self.rttm_files[i:i+step] for i in range(0, len(self.rttm_files),step)]
        return params
    
    def load_data_parallel(self):
        params = self.gen_params()
        p = multiprocessing.Pool(processes=len(params))
        print("-------- load data parallel----------")
        result = p.map(self.load_data, params)
        
        p.close()
        p.join()
        print("---------- done ----------")
        
        for index in tqdm(range(len(result))):
            _dict = result[index]
            for key, value in _dict.items():
                self.audios[key] = value
            _dict = None
        print("total file: ", len(self.audios.keys()))
        
    
    def load_data(self, rttm_files):
        audios = {}
        for rttm_path in tqdm(rttm_files):
            file = rttm_path.split("/")[-1]
            audio_name = file.split(".")[0]
            
            for audio_path in self.wav_files:
                if audio_name != audio_path.split("/")[-3]:
                # if audio_name not in audio_path:
                    continue
                
                audios[audio_name] = Audio(
                        rttm_path=rttm_path, 
                        audio_path=audio_path,
                        sample_rate=self.sample_rate
                    )

        return audios
        