#!/usr/bin/env python3
"""
Keep System Awake Script
------------------------
Prevents system sleep during long-running tasks like ML training.

Features:
- Multiple wake-up methods (mouse movement, key press, screen saver prevention)
- Configurable intervals and methods
- Better error handling and logging
- System-specific optimizations
- Graceful shutdown
"""

import time
import sys
import os
import platform
from datetime import datetime
import threading
import signal

# Try to import pyautogui, but don't fail if not available
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    print("⚠️  pyautogui not available - using alternative methods")

# Try to import ctypes for Windows-specific features
try:
    import ctypes
    from ctypes import wintypes
    CTYPES_AVAILABLE = True
except ImportError:
    CTYPES_AVAILABLE = False

class KeepAwake:
    def __init__(self, interval=30, method='mouse', verbose=True):
        """
        Initialize keep-awake system
        
        Args:
            interval: Seconds between wake-up actions
            method: 'mouse', 'key', 'screen', or 'all'
            verbose: Print status messages
        """
        self.interval = interval
        self.method = method
        self.verbose = verbose
        self.running = False
        self.start_time = None
        
        # Windows-specific settings
        self.is_windows = platform.system() == 'Windows'
        self.original_screen_saver = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def _prevent_screen_saver(self):
        """Prevent screen saver on Windows"""
        if not self.is_windows or not CTYPES_AVAILABLE:
            return False
        
        try:
            # Set thread execution state to prevent sleep
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            )
            return True
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Could not prevent screen saver: {e}")
            return False
    
    def _restore_screen_saver(self):
        """Restore normal screen saver behavior"""
        if not self.is_windows or not CTYPES_AVAILABLE:
            return
        
        try:
            ES_CONTINUOUS = 0x80000000
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Could not restore screen saver: {e}")
    
    def _move_mouse(self):
        """Move mouse slightly to prevent sleep"""
        if not PYAUTOGUI_AVAILABLE:
            return False
        
        try:
            # Get current position
            current_x, current_y = pyautogui.position()
            
            # Move mouse in a small circle pattern
            movements = [
                (1, 0), (0, 1), (-1, 0), (0, -1)  # Small square pattern
            ]
            
            for dx, dy in movements:
                pyautogui.moveRel(dx, dy, duration=0.1)
                time.sleep(0.05)
            
            # Return to original position
            pyautogui.moveTo(current_x, current_y, duration=0.1)
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Mouse movement failed: {e}")
            return False
    
    def _press_key(self):
        """Press a harmless key to prevent sleep"""
        if not PYAUTOGUI_AVAILABLE:
            return False
        
        try:
            # Press and release Scroll Lock (harmless key)
            pyautogui.press('scrolllock')
            time.sleep(0.1)
            pyautogui.press('scrolllock')  # Toggle back
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Key press failed: {e}")
            return False
    
    def _perform_wake_action(self):
        """Perform the configured wake-up action"""
        success = False
        
        if self.method == 'mouse' or self.method == 'all':
            if self._move_mouse():
                success = True
                if self.verbose:
                    print("🖱️  Mouse movement")
        
        if self.method == 'key' or self.method == 'all':
            if self._press_key():
                success = True
                if self.verbose:
                    print("⌨️  Key press")
        
        if self.method == 'screen' or self.method == 'all':
            if self._prevent_screen_saver():
                success = True
                if self.verbose:
                    print("🖥️  Screen saver prevention")
        
        if not success:
            if self.verbose:
                print("⚠️  No wake-up methods available")
        
        return success
    
    def _get_elapsed_time(self):
        """Get elapsed time since start"""
        if not self.start_time:
            return 0
        return time.time() - self.start_time
    
    def _format_duration(self, seconds):
        """Format duration in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def start(self):
        """Start the keep-awake system"""
        self.running = True
        self.start_time = time.time()
        
        print("🚀 Starting keep-awake system...")
        print(f"📋 Configuration:")
        print(f"   Method: {self.method}")
        print(f"   Interval: {self.interval} seconds")
        print(f"   Platform: {platform.system()}")
        print(f"   pyautogui: {'✅ Available' if PYAUTOGUI_AVAILABLE else '❌ Not available'}")
        print(f"   ctypes: {'✅ Available' if CTYPES_AVAILABLE else '❌ Not available'}")
        print()
        print("💡 Press Ctrl+C to stop")
        print("=" * 50)
        
        # Prevent screen saver immediately
        if self.method in ['screen', 'all']:
            self._prevent_screen_saver()
        
        try:
            while self.running:
                # Perform wake action
                if self._perform_wake_action():
                    elapsed = self._get_elapsed_time()
                    if self.verbose:
                        print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - "
                              f"Running for {self._format_duration(elapsed)}")
                
                # Wait for next interval
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the keep-awake system"""
        if not self.running:
            return
        
        self.running = False
        
        # Restore screen saver
        if self.method in ['screen', 'all']:
            self._restore_screen_saver()
        
        # Calculate total runtime
        elapsed = self._get_elapsed_time()
        print(f"\n✅ Keep-awake stopped")
        print(f"📊 Total runtime: {self._format_duration(elapsed)}")
        print("🔄 System settings restored")


def main():
    """Main function with command-line argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Keep system awake during long-running tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python keep_awake.py                    # Default: mouse movement every 30s
  python keep_awake.py -i 60 -m all      # All methods every 60s
  python keep_awake.py -m screen -v       # Screen saver prevention only
  python keep_awake.py -i 15 -m key      # Key press every 15s
        """
    )
    
    parser.add_argument(
        '-i', '--interval',
        type=int,
        default=30,
        help='Interval between wake actions in seconds (default: 30)'
    )
    
    parser.add_argument(
        '-m', '--method',
        choices=['mouse', 'key', 'screen', 'all'],
        default='mouse',
        help='Wake-up method (default: mouse)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    args = parser.parse_args()
    
    # Handle quiet mode
    if args.quiet:
        args.verbose = False
    
    # Validate interval
    if args.interval < 5:
        print("⚠️  Warning: Very short intervals may cause performance issues")
    elif args.interval > 3600:
        print("⚠️  Warning: Very long intervals may not prevent sleep effectively")
    
    # Create and start keep-awake system
    keep_awake = KeepAwake(
        interval=args.interval,
        method=args.method,
        verbose=args.verbose
    )
    
    keep_awake.start()


if __name__ == "__main__":
    main()