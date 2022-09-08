import torchaudio


def load_wave(file_name: str):
    data, fs = torchaudio.load(file_name, normalize=True)
    return data, fs

def metadata_wave(file_name: str):
    metadata = torchaudio.info(file_name)
    params = [
        'num_frames', 'num_channels', 'sample_rate', 'bits_per_sample', 'encoding'
    ]
    return {param: getattr(metadata, param) for param in params}
