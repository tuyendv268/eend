import soundfile as sf
import torch
import librosa
import numpy as np
import pandas as pd

def load_wav(path, start=0, end=None):
    data, samplerate = sf.read(path, start=start, stop=end)
    return data, samplerate

def count_frames(data_len, size, step):
    return int((data_len - size + step) / step)

def stft(
        data,
        frame_size=1024,
        frame_shift=256):
    fft_size = 1 << (frame_size-1).bit_length()

    if len(data) % frame_shift == 0:
        return librosa.stft(data, n_fft=fft_size, win_length=frame_size,
                            hop_length=frame_shift).T[:-1]
    else:
        return librosa.stft(data, n_fft=fft_size, win_length=frame_size,
                            hop_length=frame_shift).T

def gen_frame_indices(
    data_length, 
    size=2000, 
    step=2000, 
    use_last_samples=False,
    label_delay=0, 
    subsampling=1):
    
    for i in range(count_frames(data_len=data_length, size=size, step=step)):
        yield i * step, i*step+size
    if use_last_samples and i*step + size < data_length:
        if data_length - (i + 1) * step - subsampling * label_delay > 0:
            yield (i + 1) * step, data_length

def transform(
        Y,
        transform_type=None,
        dtype=np.float32):
    """ Transform STFT feature

    Args:
        Y: STFT
            (n_frames, n_bins)-shaped np.complex array
        transform_type:
            None, "log"
        dtype: output data type
            np.float32 is expected
    Returns:
        Y (numpy.array): transformed feature
    """
    Y = np.abs(Y)
    if not transform_type:
        pass
    elif transform_type == 'log':
        Y = np.log(np.maximum(Y, 1e-10))
    elif transform_type == 'logmel':
        n_fft = 2 * (Y.shape[1] - 1)
        sr = 16000
        n_mels = 40
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels)
        # Y = np.dot(Y ** 2, mel_basis.T)
        # Y = np.log10(np.maximum(Y, 1e-10))
        
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        # print("device: ", device)
        tmp_1 = torch.tensor(Y ** 2, device=device, dtype=torch.float32)
        tmp_2 = torch.tensor(mel_basis.T, device=device, dtype=torch.float32)
        Y = torch.mm(tmp_1, tmp_2)
        
        Y = torch.log10(torch.maximum(Y, torch.tensor(1e-10, device=device)))
        Y = Y.cpu().numpy()
    elif transform_type == 'logmel23':
        n_fft = 2 * (Y.shape[1] - 1)
        sr = 8000
        n_mels = 23
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels)
        Y = np.dot(Y ** 2, mel_basis.T)
        Y = np.log10(np.maximum(Y, 1e-10))
    elif transform_type == 'logmel23_mn':
        n_fft = 2 * (Y.shape[1] - 1)
        sr = 8000
        n_mels = 23
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels)
        Y = np.dot(Y ** 2, mel_basis.T)
        
        # device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        # print("device: ", device)
        # tmp_1 = torch.tensor(Y ** 2, device=device, dtype=torch.float)
        # tmp_2 = torch.tensor(mel_basis.T, device=device, dtype=torch.float)
        # Y = torch.mm(tmp_1, tmp_2)
        
        # Y = torch.log10(torch.maximum(Y, torch.tensor(1e-10, device=device)))
        # mean = torch.mean(Y, dim=0)
        
        Y = np.log10(np.maximum(Y, 1e-10))
        mean = np.mean(Y, axis=0)
        Y = Y - mean
        Y = Y.cpu().numpy()
    elif transform_type == 'logmel23_swn':
        n_fft = 2 * (Y.shape[1] - 1)
        sr = 8000
        n_mels = 23
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels)
        Y = np.dot(Y ** 2, mel_basis.T)
        Y = np.log10(np.maximum(Y, 1e-10))
        #b = np.ones(300)/300
        #mean = scipy.signal.convolve2d(Y, b[:, None], mode='same')
        #
        # simple 2-means based threshoding for mean calculation
        powers = np.sum(Y, axis=1)
        th = (np.max(powers) + np.min(powers))/2.0
        for i in range(10):
            th = (np.mean(powers[powers >= th]) + np.mean(powers[powers < th])) / 2
        mean = np.mean(Y[powers > th,:], axis=0)
        Y = Y - mean
    elif transform_type == 'logmel23_mvn':
        n_fft = 2 * (Y.shape[1] - 1)
        sr = 8000
        n_mels = 23
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels)
        Y = np.dot(Y ** 2, mel_basis.T)
        Y = np.log10(np.maximum(Y, 1e-10))
        mean = np.mean(Y, axis=0)
        Y = Y - mean
        std = np.maximum(np.std(Y, axis=0), 1e-10)
        Y = Y / std
    else:
        raise ValueError('Unknown transform_type: %s' % transform_type)
    return Y.astype(dtype)

def subsample(Y, T, subsampling=1):
    """ Frame subsampling
    """
    Y_ss = Y[::subsampling]
    T_ss = T[::subsampling]
    return Y_ss, T_ss


def splice(Y, context_size=0):
    """ Frame splicing

    Args:
        Y: feature
            (n_frames, n_featdim)-shaped numpy array
        context_size:
            number of frames concatenated on left-side
            if context_size = 5, 11 frames are concatenated.

    Returns:
        Y_spliced: spliced feature
            (n_frames, n_featdim * (2 * context_size + 1))-shaped
    """
    Y_pad = np.pad(
            Y,
            [(context_size, context_size), (0, 0)],
            'constant')
    Y_spliced = np.lib.stride_tricks.as_strided(
            Y_pad,
            (Y.shape[0], Y.shape[1] * (2 * context_size + 1)),
            (Y.itemsize * Y.shape[1], Y.itemsize), writeable=False)
    return Y_spliced

def get_labeledSTFT(audio,
                    start, 
                    end, 
                    frame_size, 
                    frame_shift, 
                    n_speakers=None):
    # path = "amicorpus/wav/ES2002a/audio/ES2002a.Mix-Headset.wav"
    # datas, rate = load_wav(path, start*frame_shift, end*frame_shift)
    datas, rate = audio.load_wav(start = start*frame_shift, end=end*frame_shift)
    # print(datas.shape)
    # print(rate)
    Y = stft(datas, frame_size, frame_shift)
    
    # print(f"start_time: {start*frame_shift/rate}")
    # print(f"end_time: {end*frame_shift/rate}")
    a = (audio.metadata["st"] >= start*frame_shift/rate) & (audio.metadata["et"] < end*frame_shift/rate)
    b = (audio.metadata["st"] < end*frame_shift/rate) & (audio.metadata["et"] >= end*frame_shift/rate)
    
    segments = audio.metadata[a|b]
    # print(f"segments: \n{segments}")
    speaker_ids = segments["spk_id"].unique().tolist()
    
    # if len(speaker_ids) > 2:
        # return None, None
        
    # print(speaker_ids)

    if n_speakers == None:
        n_speakers = len(speaker_ids)
    # print(n_speakers)
    # print(f"Y shape: {Y.shape}")
    T = np.zeros((Y.shape[0], n_speakers), dtype=np.int8)
    
    for index in segments.index:
        # print(f'spk_id: {segments["spk_id"][index]}')
        speaker_idx = speaker_ids.index(segments["spk_id"][index])
        # print(f'speaker_index: {speaker_idx}')
        
        start_frame = np.rint(segments["st"][index] * rate / frame_shift).astype(int) - start
        end_frame = np.rint(segments["et"][index] * rate / frame_shift).astype(int) - start
        # print(f"start_frame: {start_frame}")
        # print(f"end_frame: {end_frame}")
        
        T[start_frame:end_frame, speaker_idx] = 1
        # print(T[start_frame:end_frame, speaker_idx])
    return Y, T        