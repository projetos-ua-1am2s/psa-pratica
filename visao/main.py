import cv2
import sys
from pathlib import Path

try:
    # Prefer relative import when used as a package
    from .person_tracker import PersonTracker
except ImportError:
    # Fallback for running this file directly as a script
    from person_tracker import PersonTracker


def main():
    # 1. Initialize System
    data_config_path = Path(__file__).resolve().parent / "data.yaml"
    surveillance_system = PersonTracker(conf_threshold=0.5)

    print("\n--- Surveillance System Active ---")
    print("Commands:")
    print("- Press 'Ctrl+C' in the terminal to stop.")
    print("- Press 'q' while focused on the video window to quit.")
    print("----------------------------------\n")

    try:
        # 2. Iterate over the generator sequentially
        # The generator 'yields' data as fast as the AI processes it.
        for vector, frame in surveillance_system.run():

            if vector:
                magnitude, angle = vector
                # Your camera movement logic goes here
                # Example: print only if the movement is significant
                if magnitude > 0.05:
                    print(f"[ACTION] Tracking: Dist={magnitude:.3f}, Angle={angle:.1f}°")

            # Show the result visually
            cv2.imshow("Tracking View", frame)

            # Check for 'q' key to quit via OpenCV window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuit requested via keyboard (q).")
                break

    except KeyboardInterrupt:
        # 3. Handle Ctrl+C gracefully
        print("\n\n[INTERRUPT] Stop signal received (Ctrl+C). Cleaning up...")

    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")

    finally:
        # 4. Ensure resources are released
        surveillance_system.cleanup()
        print("System shutdown complete.")


if __name__ == "__main__":
    main()