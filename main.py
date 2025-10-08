import cv2
import argparse
import time
from src.pipeline.processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description='Tram CV System - Object Detection and Distance Estimation')
    parser.add_argument('--source', type=str, help='Video source (path to video file)')
    parser.add_argument('--verbose', type=bool, default=True)
    args = parser.parse_args()
    
    processor = VideoProcessor()
    processor.process_video(args.source, args.verbose)

if __name__ == "__main__":
    main()