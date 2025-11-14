#!/usr/bin/env python3
"""
ZORDIC SENSORIUM Launcher
Real-time AI HUD interface
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sensorium_hud import ZordicSensorium

def main():
    """Launch sensorium"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║              ZORDIC SENSORIUM - AI HUD Interface                ║
║                                                                  ║
║            Real-time Visualization of AI Internal State         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    print("Initializing sensorium components...")
    print("  ✓ φ-field visualizer")
    print("  ✓ Token stream processor")
    print("  ✓ Regime state monitor")
    print("  ✓ Metrics panel")
    print()
    print("Launching HUD...")
    print()
    print("Controls:")
    print("  ESC - Exit")
    print("  SPACE - Pause/Resume")
    print()

    try:
        sensorium = ZordicSensorium()
        sensorium.run()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\nSensorium terminated")


if __name__ == "__main__":
    main()
