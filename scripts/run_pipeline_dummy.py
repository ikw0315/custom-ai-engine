import numpy as np
from src.pipeline import PipelineManager
from src.video_parser import VideoParser

class DummyVideoParser(VideoParser):
    """
    A dummy video parser class for testing the pipeline without actual video files.
    Generates random frames with dummy text and action patterns.
    """
    def __init__(self, num_frames: int = 100):
        self.frame_count = num_frames
        self.fps = 30.0

    def extract_frame_data(self, max_frames: int = 100):
        frame_data = []
        for i in range(min(self.frame_count, max_frames)):
            # Dummy frame: random 64x64 RGB image
            visual_feature = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            text_element = f"Frame {i}: dummy text analysis"
            action_pattern = {'objects_detected': np.random.randint(1, 10), 'dominant_color': 'gray'}

            frame_data.append({
                'frame_index': i,
                'visual_feature': visual_feature,
                'text_element': text_element,
                'action_pattern': action_pattern
            })
        print(f"Generated {len(frame_data)} dummy frames for testing.")
        return frame_data

# Use DummyVideoParser in the existing PipelineManager
pipeline = PipelineManager(video_path=None)  # video_path is None because we use dummy frames
pipeline.parser = DummyVideoParser(num_frames=50)
pipeline.run_automated_pipeline(max_frames_to_process=50, do_retrain=False)
