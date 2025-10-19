from src.video_parser import VideoParser
from src.engine import Engine

class PipelineManager:
    """
    Automates the video parsing, AI learning/analysis, and retraining 
    pipeline to run with minimal human intervention.
    """
    def __init__(self, video_path: str, model_state_path: str = "engine_state.json"):
        self.video_path = video_path
        self.model_state_path = model_state_path
        self.parser = VideoParser(video_path)
        self.engine = Engine()

    def run_automated_pipeline(self, max_frames_to_process: int = 100, do_retrain: bool = True):
        """
        Execute the full automated pipeline: video analysis, learning, and result recording.
        """
        print("="*50)
        print("Starting Automated AI Video Analysis Pipeline")
        print("="*50)

        # 1. Attempt to load previous model state (for retraining)
        try:
            self.engine.load_model(self.model_state_path)
        except FileNotFoundError:
            print("No previous model found. Starting with initial model.")

        # 2. Extract and preprocess video frames (VideoParser role)
        print("\n--- Phase 1: Video Parsing ---")
        try:
            raw_data = self.parser.extract_frame_data(max_frames=max_frames_to_process)
        except IOError as e:
            print(f"Pipeline failed during parsing: {e}")
            return

        # 3. AI analysis and learning (Engine role)
        print("\n--- Phase 2: AI Analysis & Learning ---")
        analysis_result = self.engine.analyze_and_learn(raw_data)

        # 4. Record learning result and prepare for retraining
        print("\n--- Phase 3: Result Recording & State Saving ---")
        self.engine.save_state(self.model_state_path)
        
        if do_retrain and analysis_result['loss_after_training'] > 0.3:
            print("Warning: Loss is high (0.3 threshold). Initiating automatic RETRAINING cycle.")
            # In practice, call data queuing and distributed retraining logic
        else:
            print("Analysis complete. System ready for next task.")
