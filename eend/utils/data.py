import os
from glob import glob
import logging

from eend.utils.audio import Audio


class Data():
    def __init__(self, path, sample_rate):
        self.path = path
        self.sample_rate = sample_rate
        rttm_path = os.path.join(path, "rttm/*.rttm")
        audio_path = os.path.join(path, "wav/*/audio/*")
        
        self.rttm_files = glob(rttm_path)
        self.wav_files = glob(audio_path)
        self.audios = {}
        
        self.load_data()
        
    def load_data(self):
        logger = logging.getLogger(__name__)
        logger.info("-------- load data ----------")
        for rttm_path in self.rttm_files:
            file = rttm_path.split("/")[-1]
            audio_name = file.split(".")[0]
            
            for audio_path in self.wav_files:
                if audio_name not in audio_path:
                    continue
                
                self.audios[audio_name] = Audio(
                        rttm_path=rttm_path, 
                        audio_path=audio_path,
                        sample_rate=self.sample_rate
                    )

        logger.info("---------- done ----------")
        
        