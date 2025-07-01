# import tkinter as tk
# from gui.main_window import MainApp

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = MainApp(root)
#     root.mainloop()


#!/usr/bin/env python3
"""
Main entry point for the Face Recognition Attendance System GUI
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the GUI application
from gui.main_window import main

if __name__ == "__main__":
    main()