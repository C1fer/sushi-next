import logging
import wave
import numpy as np
from time import time
from scipy.signal import resample

from .common import SushiError, clip


class DownmixedWavFile:
    """
    Reads a WAV file and provides downmixed mono frames as float32.
    """
    def __init__(self, path):
        try:
            self._wave = wave.open(path, 'rb')
        except wave.Error as e:
            raise SushiError(f"Cannot open WAV file: {e}")

        self.channels_count = self._wave.getnchannels()
        self.framerate = self._wave.getframerate()
        self.sample_width = self._wave.getsampwidth()
        self.frames_count = self._wave.getnframes()
        self.frame_size = self.channels_count * self.sample_width

    def close(self):
        if self._wave:
            self._wave.close()
            self._wave = None

    def readframes(self, count):
        if count <= 0:
            return np.array([], dtype=np.float32)

        raw = self._wave.readframes(count)
        if self.sample_width == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        else:
            raise SushiError(f"Unsupported sample width: {self.sample_width}")

        if self.channels_count > 1:
            # reshape to (n_frames, channels) and average
            data = data.reshape(-1, self.channels_count)
            data = data.mean(axis=1)

        return data


class WavStream:
    """
    Loads a WAV file into memory, downmixes to mono, resamples,
    normalizes and provides windowed access and pattern matching.
    """
    READ_CHUNK_SIZE = 1       # seconds per read
    PADDING_SECONDS = 10      # padding on each side (seconds)

    def __init__(self, path, sample_rate=12000, sample_type='uint8'):
        if sample_type not in ('float32', 'uint8'):
            raise SushiError('Unknown sample type of WAV stream, must be uint8 or float32')

        stream = DownmixedWavFile(path)
        total_seconds = stream.frames_count / float(stream.framerate)
        downsample_ratio = sample_rate / float(stream.framerate)

        self.sample_rate = sample_rate
        self.sample_count = int(np.ceil(total_seconds * sample_rate))
        pad_samples = int(self.PADDING_SECONDS * stream.framerate)
        total_samples = pad_samples * 2 + self.sample_count

        data = np.empty(total_samples, dtype=np.float32)
        pos = pad_samples
        seconds_read = 0.0
        start_time = time()

        while seconds_read < total_seconds:
            frames_to_read = int(self.READ_CHUNK_SIZE * stream.framerate)
            chunk = stream.readframes(frames_to_read)

            if downsample_ratio != 1.0:
                new_len = int(round(len(chunk) * downsample_ratio))
                chunk = resample(chunk, new_len)

            data[pos:pos + len(chunk)] = chunk
            pos += len(chunk)
            seconds_read += self.READ_CHUNK_SIZE

        # pad both ends with edge values
        data[:pad_samples] = data[pad_samples]
        data[pos:pos + pad_samples] = data[pos - 1]

        # normalize and clip
        max_val = np.median(data[data >= 0]) * 3
        min_val = np.median(data[data <= 0]) * 3
        np.clip(data, min_val, max_val, out=data)
        data = (data - min_val) / (max_val - min_val)

        if sample_type == 'uint8':
            data = (data * 255.0 + 0.5).astype(np.uint8)

        self.data = data[np.newaxis, :]
        stream.close()
        logging.info(f"Done reading WAV {path} in {time() - start_time}s")

    @property
    def duration_seconds(self):
        return self.sample_count / float(self.sample_rate)

    def get_substream(self, start, end):
        start_idx = self._get_sample_for_time(start)
        end_idx = self._get_sample_for_time(end)
        return self.data[:, start_idx:end_idx]

    def _get_sample_for_time(self, timestamp):
        return int(self.sample_rate * timestamp) + int(self.PADDING_SECONDS * self.sample_rate)

    def find_substream(self, pattern, window_center, window_size):
        start_time = clip(window_center - window_size,
                          -self.PADDING_SECONDS, self.duration_seconds)
        end_time = clip(window_center + window_size,
                        0, self.duration_seconds + self.PADDING_SECONDS)

        start_idx = self._get_sample_for_time(start_time)
        end_idx = self._get_sample_for_time(end_time) + pattern.shape[1]

        search_src = self.data[:, start_idx:end_idx]
        # use OpenCV template matching for pattern search
        import cv2
        result = cv2.matchTemplate(search_src, pattern, cv2.TM_SQDIFF_NORMED)
        min_val, _, min_loc, _ = cv2.minMaxLoc(result)
        offset = min_loc[0]
        return min_val, start_time + offset / float(self.sample_rate)
