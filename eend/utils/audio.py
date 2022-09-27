import pandas as pd
# import soundfile as sf
import librosa
from functools import lru_cache

class Audio():
    def __init__(self, rttm_path, audio_path, sample_rate):
        self.sample_rate = sample_rate
        self.audio_path = audio_path
        # self.length = self.load()
        
        self.metadata = pd.read_csv(rttm_path, sep=" ", header=None)
        self.metadata.columns=["utt","rec","sth_1","st","dur","sth_2","sth_3","spk_id","sth_4","sth_5", "et"]
        
        # self.metadata["et"] = self.metadata["st"] + self.metadata["dur"]
        self.data, self.samplerate = librosa.load(self.audio_path, sr=self.sample_rate)
        self.length = len(self.data)
    
    @lru_cache(maxsize=1)
    def load_wav(self, start=0, end=None):
        # data, samplerate = sf.read(self.audio_path, start=start, stop=end)
        # data, samplerate = librosa.load(self.audio_path, sr=self.sample_rate)
        return self.data[start:end], self.samplerate
    
    def load(self):
        data, samplerate = librosa.load(self.audio_path, sr=self.sample_rate)
        return len(data)
        