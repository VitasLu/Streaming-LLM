import torch
import torchaudio
import numpy as np
from typing import Tuple, Iterator, Optional

class SegmentWrapper:
    def __init__(self, audio_path: str, segment_length: float, sample_rate: int = 16000):
        """Initialize a streaming audio segment wrapper.
        
        Args:
            audio_path: Path to the audio file
            segment_length: Length of each segment in seconds
            sample_rate: Sample rate of the audio
        """
        self.audio_path = audio_path
        self.sample_rate = sample_rate
        
        # Calculate frame parameters
        self.samples_to_read = int(segment_length * sample_rate)
        # Add some overlap to smooth transitions between segments
        self.overlap_samples = int(0.5 * sample_rate)  # 500ms overlap
        self.samples_in_chunk = self.samples_to_read + self.overlap_samples
        
        # Load audio
        audio, sr = torchaudio.load(audio_path, normalize=True)
        assert sr == sample_rate, f"Expected sample rate {sample_rate}, got {sr}"
        self.audio = audio.squeeze()
        self.audio_len_s = self.audio.shape[0] / sample_rate
        self.total_segments = int(np.ceil(self.audio.shape[0] / self.samples_to_read))
        
        print(f"Loaded audio of length {self.audio_len_s:.2f}s, will process in {self.total_segments} segments")

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, bool]]:
        """Iterate through audio segments.
        
        Yields:
            Tuple containing:
                - Audio segment tensor
                - Boolean indicating if this is the last segment
        """
        # First segment
        if self.audio.shape[0] <= self.samples_in_chunk:
            # Audio shorter than segment length, pad if needed
            frames_in_chunk = torch.zeros(self.samples_in_chunk)
            frames_in_chunk[:self.audio.shape[0]] = self.audio
            yield frames_in_chunk, True
            return
            
        frames_in_chunk = self.audio[:self.samples_in_chunk]
        read_pointer = self.samples_to_read  # Move pointer by segment length, not including overlap
        yield frames_in_chunk, (read_pointer >= self.audio.shape[0])
        
        # Subsequent segments
        while read_pointer < self.audio.shape[0]:
            # Include overlap from previous segment
            start_idx = read_pointer - self.overlap_samples
            end_idx = min(read_pointer + self.samples_to_read, self.audio.shape[0])
            
            # Create chunk with consistent length
            frames_in_chunk = torch.zeros(self.samples_in_chunk)
            actual_len = end_idx - start_idx
            frames_in_chunk[:actual_len] = self.audio[start_idx:end_idx]
            
            read_pointer += self.samples_to_read
            is_last = (read_pointer >= self.audio.shape[0])
            
            yield frames_in_chunk, is_last

class StreamingAudioProcessor:
    def __init__(
        self, 
        audio_path: str, 
        segment_length: float = 5.0,
        sample_rate: int = 16000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the streaming audio processor.
        
        Args:
            audio_path: Path to audio file
            segment_length: Length of segments in seconds
            sample_rate: Audio sample rate
            device: Device to run processing on
        """
        self.segment_wrapper = SegmentWrapper(audio_path, segment_length, sample_rate)
        self.device = device
        self.sample_rate = sample_rate
        
        # State variables
        self.segment_count = 0
        self.is_finished = False
        
    def get_segments(self):
        """Get audio segments for processing.
        
        Yields:
            dict: Segment information with keys:
                - 'frames': Audio frames tensor
                - 'is_last': Whether this is the last segment
                - 'segment_id': Current segment number
        """
        for frames, is_last in self.segment_wrapper:
            self.segment_count += 1
            
            yield {
                'frames': frames.to(self.device),
                'is_last': is_last,
                'segment_id': self.segment_count
            }
            
            if is_last:
                self.is_finished = True

    def get_total_segments(self):
        """Get the total number of segments in the audio file."""
        return self.segment_wrapper.total_segments
    
    def get_audio_length(self):
        """Get the total audio length in seconds."""
        return self.segment_wrapper.audio_len_s
    
    def reset(self):
        """Reset the processor state."""
        self.segment_count = 0
        self.is_finished = False