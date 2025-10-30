#!/usr/bin/env python3
"""
ZORDIC Desktop Launcher
Leviathan AI Corporation

Launch the ZORDIC φ-field self-organizing lattice application
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import tkinter as tk
    from zordic_gui import ZordicDesktopApp
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("\nPlease install requirements:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def main():
    """Main entry point"""
    print("="*70)
    print("  ZORDIC - φ-Field Self-Organizing Lattice System")
    print("  Leviathan AI Corporation")
    print("="*70)
    print()
    print("Initializing application...")
    print()

    try:
        root = tk.Tk()
        app = ZordicDesktopApp(root)

        print("✓ Application started successfully")
        print("  Window should appear now...")
        print()

        root.mainloop()

    except KeyboardInterrupt:
        print("\n\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
