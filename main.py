"""
Motion Detection System - Main Program
=======================================

Sistem deteksi gerak menggunakan classical computer vision.
Metode: Running Average Background Subtraction

Fitur:
- Webcam atau video file input
- Motion tracking dengan trails
- Motion heatmap visualization
- Recording dan snapshot
- Audio alerts

Cara menjalankan:
    python main.py --video path/to/video.mp4    # Dari video file
    python main.py --webcam                      # Dari webcam
    python main.py --webcam 1                    # Webcam index 1
    
Keyboard Shortcuts:
    Q       - Keluar
    D       - Toggle debug view
    H       - Toggle heatmap
    T       - Toggle tracking trails
    S       - Ambil snapshot
    R       - Start/stop recording
    Space   - Pause/resume
    +/-     - Adjust threshold
    L       - Toggle keyboard legend
"""

import argparse
import time
import cv2
import numpy as np
from typing import Optional

from src.video_reader import VideoReader
from src.preprocessing import preprocess_frame
from src.background_model import create_background_model
from src.motion_detector import MotionDetector
from src.visualization import (
    draw_motion_regions,
    create_debug_view,
    add_status_bar,
    add_keyboard_legend,
    add_progress_bar,
    draw_center_message
)
from src.motion_tracker import MotionTracker, draw_tracking
from src.heatmap import MotionHeatmap
from src.recorder import VideoRecorder, SnapshotCapture
from src.alert import MotionAlert


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Motion Detection System - Enhanced Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python main.py --video video.mp4
  python main.py --webcam
  python main.py --webcam 1 --record
  python main.py --video video.mp4 --threshold 30 --min-area 1000
        """
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video", "-v", type=str,
        help="Path ke file video"
    )
    input_group.add_argument(
        "--webcam", "-w", nargs="?", const=0, type=int,
        help="Gunakan webcam (default: 0)"
    )
    
    # Detection parameters
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
    
    # Feature toggles
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Mulai dengan debug view aktif"
    )
    parser.add_argument(
        "--tracking", action="store_true",
        help="Aktifkan motion tracking"
    )
    parser.add_argument(
        "--heatmap", action="store_true",
        help="Aktifkan motion heatmap"
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Langsung mulai recording"
    )
    parser.add_argument(
        "--alert", action="store_true",
        help="Aktifkan audio alert saat gerakan"
    )
    parser.add_argument(
        "--no-legend", action="store_true",
        help="Sembunyikan keyboard legend"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Direktori untuk menyimpan recording/snapshot"
    )
    
    return parser.parse_args()


def print_banner(args):
    """Print startup banner."""
    print("=" * 60)
    print("  MOTION DETECTION SYSTEM - Enhanced Edition")
    print("=" * 60)
    
    if args.video:
        print(f"  📹 Source: Video file - {args.video}")
    else:
        print(f"  📷 Source: Webcam #{args.webcam}")
    
    print(f"  🔧 Method: {args.method}")
    print(f"  📊 Threshold: {args.threshold}")
    print(f"  📐 Min Area: {args.min_area} px²")
    
    features = []
    if args.tracking:
        features.append("Tracking")
    if args.heatmap:
        features.append("Heatmap")
    if args.record:
        features.append("Recording")
    if args.alert:
        features.append("Alert")
    
    if features:
        print(f"  ✨ Features: {', '.join(features)}")
    
    print("-" * 60)
    print("  Keyboard: Q=Quit D=Debug H=Heat T=Track S=Snap R=Rec")
    print("=" * 60)


def main():
    """Main function - menjalankan motion detection."""
    args = parse_arguments()
    print_banner(args)
    
    # === Inisialisasi komponen ===
    
    # Background model
    bg_kwargs = {"alpha": args.alpha} if args.method == "running_average" else {}
    bg_model = create_background_model(args.method, **bg_kwargs)
    
    # Motion detector
    detector = MotionDetector(
        threshold=args.threshold,
        min_area=args.min_area
    )
    
    # Motion tracker
    tracker = MotionTracker(
        max_disappeared=30,
        max_distance=100,
        trail_length=30
    )
    
    # Heatmap
    heatmap = MotionHeatmap(
        decay_rate=0.995,
        intensity=5.0
    )
    
    # Recorder & Snapshot
    recorder = VideoRecorder(output_dir=args.output_dir)
    snapshot = SnapshotCapture(output_dir=args.output_dir)
    
    # Alert
    alert = MotionAlert(cooldown_seconds=2.0, enabled=args.alert)
    
    # === State variables ===
    show_debug = args.debug
    show_tracking = args.tracking
    show_heatmap = args.heatmap
    show_legend = not args.no_legend
    paused = False
    current_threshold = args.threshold
    
    frame_times = []
    frame_number = 0
    total_frames = 0
    
    # Determine input source
    source = args.video if args.video else args.webcam
    
    try:
        with VideoReader(source) as reader:
            info = reader.get_info()
            print(f"\n📺 Resolution: {info['width']}x{info['height']}")
            print(f"🎬 FPS: {info['fps']:.2f}")
            if not reader.is_webcam:
                print(f"⏱️  Duration: {info['duration_seconds']:.2f}s")
                total_frames = info['frame_count']
            print()
            
            # Start recording jika diminta
            if args.record:
                recorder.start((info['width'], info['height']))
            
            for frame in reader.frames():
                frame_number += 1
                
                # Handle pause
                if paused:
                    display = draw_center_message(frame, "PAUSED", "Press SPACE to resume")
                    cv2.imshow("Motion Detection", display)
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord(' '):
                        paused = False
                    elif key == ord('q'):
                        break
                    continue
                
                start_time = time.time()
                
                # === Step 1: Preprocessing ===
                gray_blurred = preprocess_frame(frame, blur_size=args.blur_size)
                
                # === Step 2: Background subtraction ===
                diff = bg_model.get_foreground_mask(gray_blurred)
                bg_model.update(gray_blurred)
                
                # Update detector threshold if changed
                detector.threshold = current_threshold
                
                # === Step 3: Motion detection ===
                mask, regions = detector.detect(diff)
                
                # === Step 4: Update trackers ===
                if show_tracking:
                    detections = [
                        (region.center, region.bounding_box)
                        for region in regions
                    ]
                    tracked_objects = tracker.update(detections)
                
                # === Step 5: Update heatmap ===
                if show_heatmap:
                    heatmap.update(mask)
                
                # === Step 6: Alert ===
                if alert.enabled and len(regions) > 0:
                    total_area = sum(r.area for r in regions)
                    alert.trigger(len(regions))
                
                # === Step 7: Visualization ===
                result = frame.copy()
                
                # Heatmap overlay
                if show_heatmap:
                    result = heatmap.overlay_on_frame(result, alpha=0.4)
                
                # Motion regions
                result = draw_motion_regions(result, regions)
                
                # Tracking
                if show_tracking:
                    result = draw_tracking(
                        result, tracked_objects,
                        show_trails=True, show_id=True
                    )
                
                # Calculate FPS
                elapsed = time.time() - start_time
                frame_times.append(elapsed)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                fps = 1.0 / (sum(frame_times) / len(frame_times))
                
                # Status bar
                result = add_status_bar(
                    result, len(regions), fps,
                    is_webcam=reader.is_webcam,
                    recording=recorder.is_recording,
                    heatmap_on=show_heatmap,
                    tracking_on=show_tracking,
                    paused=paused
                )
                
                # Progress bar (untuk video file)
                if not reader.is_webcam and total_frames > 0:
                    progress = frame_number / total_frames
                    result = add_progress_bar(result, progress)
                
                # Keyboard legend
                if show_legend:
                    result = add_keyboard_legend(result)
                
                # Recording
                if recorder.is_recording:
                    recorder.write(result)
                
                # === Display ===
                if show_debug:
                    display = create_debug_view(frame, diff, mask, result)
                    cv2.imshow("Motion Detection - Debug", display)
                else:
                    cv2.imshow("Motion Detection", result)
                
                # === Handle keyboard ===
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    show_debug = not show_debug
                    cv2.destroyAllWindows()
                elif key == ord('h'):
                    show_heatmap = not show_heatmap
                    if show_heatmap:
                        print("🌡️  Heatmap ON")
                    else:
                        print("🌡️  Heatmap OFF")
                elif key == ord('t'):
                    show_tracking = not show_tracking
                    if show_tracking:
                        print("📍 Tracking ON")
                    else:
                        print("📍 Tracking OFF")
                        tracker.reset()
                elif key == ord('s'):
                    filepath = snapshot.capture(result)
                    print(f"📸 Snapshot: {filepath}")
                elif key == ord('r'):
                    if recorder.is_recording:
                        saved = recorder.stop()
                        print(f"⏹️  Recording stopped: {saved}")
                    else:
                        recorder.start((result.shape[1], result.shape[0]))
                        print("🔴 Recording started")
                elif key == ord(' '):
                    paused = True
                elif key == ord('l'):
                    show_legend = not show_legend
                elif key == ord('+') or key == ord('='):
                    current_threshold = min(255, current_threshold + 5)
                    print(f"📊 Threshold: {current_threshold}")
                elif key == ord('-'):
                    current_threshold = max(0, current_threshold - 5)
                    print(f"📊 Threshold: {current_threshold}")
                elif key == ord('a'):
                    alert.toggle()
                    print(f"🔔 Alert: {'ON' if alert.enabled else 'OFF'}")
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Dihentikan oleh user.")
    finally:
        # Cleanup
        if recorder.is_recording:
            recorder.stop()
        cv2.destroyAllWindows()
    
    print("\n✅ Selesai.")
    return 0


if __name__ == "__main__":
    exit(main())
