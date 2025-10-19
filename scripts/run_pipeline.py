from src.pipeline import PipelineManager

pipeline = PipelineManager('data/sample_video.mp4')
pipeline.run_automated_pipeline(max_frames_to_process=200, do_retrain=True)
