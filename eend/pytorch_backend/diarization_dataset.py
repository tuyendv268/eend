from torch.utils.data import Dataset
from eend.utils.utils import *
import librosa
from tqdm import tqdm

from eend.utils.data import Data

class DiarizationDataset(Dataset):
    def __init__(
            self, 
            data_path,
            chunk_size=500,
            context_size=0,
            frame_size=1024,
            frame_shift=256,
            subsampling=1,
            rate=8000,
            input_transform=None,
            use_last_samples=False,
            label_delay=0,
            n_speakers=None
        ):
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.subsampling=subsampling
        self.rate = rate
        self.input_transform = input_transform
        self.n_speakers = n_speakers
        self.label_delay = label_delay 
        self.use_last_samples = use_last_samples
        
        self.datas = Data(path=data_path, sample_rate=self.rate)
        
        self.init_chunks()
        
        self.Y, self.T = self.preprocess_data()
        self.datas = None
        
    def init_chunks(self):
        self.chunk_indices = []
        print("------------- init chunks -------------")
        for key in tqdm(self.datas.audios):
            audio = self.datas.audios[key]

            # data_len = int(audio.metadata["st"].to_numpy() * self.rate / self.frame_shift)
            data_len = int(audio.length / self.frame_shift)
            data_len = int(data_len/self.subsampling)
            
            for st, ed in gen_frame_indices(
                    data_length=data_len,
                    step=self.chunk_size,
                    size=self.chunk_size,
                    use_last_samples=self.use_last_samples,
                    label_delay=0,
                    subsampling=self.subsampling):
                self.chunk_indices.append((key, st*self.subsampling, ed*self.subsampling))
        
        self.chunk_indices = self.chunk_indices[:-1]
        print("chunks: ",len(self.chunk_indices))
        print("------------- done -------------")
    
    def preprocess_data(self):
        Y, T = [], []
        print("------------- preprocess data -------------")
        for index in tqdm(range(len(self.chunk_indices))):
            Y_ss, T_ss = self.getchunk(index)
            Y.append(tuple(Y_ss))
            T.append(tuple(T_ss))            
        print("-------------- done ---------------")

        return torch.tensor(Y).float(), torch.tensor(T).float()
    
    def __len__(self):
        return len(self.chunk_indices)
    
    def __getitem__(self, index):
        return self.Y[index], self.T[index]
    
    def getchunk(self, i):
        key, st, ed = self.chunk_indices[i]
        Y, T = get_labeledSTFT(
            self.datas.audios[key],
            st,
            ed,
            self.frame_size,
            self.frame_shift,
            self.n_speakers)
        # Y: (frame, num_ceps)
        Y = transform(Y,self.input_transform)
        # Y_spliced: (frame, num_ceps * (context_size * 2 + 1))
        Y_spliced = splice(Y, self.context_size)
        # Y_ss: (frame / subsampling, num_ceps * (context_size * 2 + 1))
        Y_ss, T_ss = subsample(Y_spliced, T, self.subsampling)

        # Y_ss = torch.from_numpy(Y_ss).float()
        # T_ss = torch.from_numpy(T_ss).float()
        return Y_ss.tolist(), T_ss.tolist()
