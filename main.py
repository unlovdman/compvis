"""
Motion Detection System - Main Program
=======================================

Sistem deteksi gerak menggunakan classical computer vision.
Metode: Running Average Background Subtraction

Cara menjalankan:
    python main.py --video path/to/video.mp4
    
Tekan 'q' untuk keluar, 'd' untuk toggle debug view.
"""

import argparse
import time
import cv2
import numpy as np

from src.video_reader import VideoReader
from src.preprocessing import preprocess_frame
from src.background_model import create_background_model
from src.motion_detector import MotionDetector
from src.visualization import (
    draw_motion_regions,
    create_debug_view,
    add_status_bar
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Motion Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python main.py --video video.mp4
  python main.py --video video.mp4 --threshold 30 --min-area 1000
  python main.py --video video.mp4 --method mog2
        """
    )
    parser.add_argument(
        "--video", "-v", type=str, required=True,
        help="Path ke file video"
    )
    parser.add_argument(
        "--method", "-m", type=str, default="running_average",
        choices=["frame_diff", "running_average", "mog2"],
        help="Metode background subtraction (default: running_average)"
    )
    parser.add_argument(
        "--threshold", "-t", type=int, default=25,
        help="Threshold untuk binarisasi (default: 25)"
    )
    parser.add_argument(
        "--min-area", "-a", type=int, default=500,
        help="Area minimum kontur dalam piksel² (default: 500)"
    )
    parser.add_argument(
        "--blur-size", "-b", type=int, default=21,
        help="Ukuran kernel Gaussian blur (default: 21)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01,
        help="Learning rate untuk running average (default: 0.01)"
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Tampilkan debug view"
    )
    
    return parser.parse_args()


def main():
    """Main function - menjalankan motion detection."""
    args = parse_arguments()
    
    print("=" * 60)
    print("MOTION DETECTION SYSTEM")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Method: {args.method}")
    print(f"Threshold: {args.threshold}")
    print(f"Min Area: {args.min_area}")
    print(f"Blur Size: {args.blur_size}")
    print("-" * 60)
    print("Tekan 'q' untuk keluar, 'd' untuk toggle debug view")
    print("=" * 60)
    
    # Inisialisasi komponen
    bg_kwargs = {"alpha": args.alpha} if args.method == "running_average" else {}
    bg_model = create_background_model(args.method, **bg_kwargs)
    detector = MotionDetector(
        threshold=args.threshold,
        min_area=args.min_area
    )
    
    show_debug = args.debug
    frame_times = []
    
    try:
        with VideoReader(args.video) as reader:
            info = reader.get_info()
            print(f"Resolution: {info['width']}x{info['height']}")
            print(f"FPS: {info['fps']:.2f}")
            print(f"Duration: {info['duration_seconds']:.2f}s")
            print("-" * 60)
            
            for frame in reader.frames():
                start_time = time.time()
                
                # Step 1: Preprocessing
                gray_blurred = preprocess_frame(frame, blur_size=args.blur_size)
                
                # Step 2: Background subtraction
                diff = bg_model.get_foreground_mask(gray_blurred)
                bg_model.update(gray_blurred)
                
                # Step 3: Motion detection
                mask, regions = detector.detect(diff)
                
                # Step 4: Visualization
                result = draw_motion_regions(frame, regions)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                frame_times.append(elapsed)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                fps = 1.0 / (sum(frame_times) / len(frame_times))
                
                # Add status bar
                result = add_status_bar(result, len(regions), fps)
                
                # Display
                if show_debug:
                    display = create_debug_view(frame, diff, mask, result)
                    cv2.imshow("Motion Detection - Debug", display)
                else:
                    cv2.imshow("Motion Detection", result)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    show_debug = not show_debug
                    cv2.destroyAllWindows()
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nDihentikan oleh user.")
    finally:
        cv2.destroyAllWindows()
    
    print("\nSelesai.")
    return 0


if __name__ == "__main__":
    exit(main())
