'''This script is designed to run a calibration sequence for a display using Tkinter. 
It shows a series of colored screens for a specified duration, with pauses in between, 
and allows the user to exit the sequence by pressing the ESC key. The colors displayed 
include neutral gray, black, white, red, green, and blue.

Data extracted will enable the user to check the calibration in terms of:
- Background Luminance
- RGB Color Accuracy
- Spatial Uniformity'''

import time
import tkinter as tk

timer = time.time()  # Start the timer

class CalibrationApp:
    def __init__(self, root, display_time=5.0, pause_between=2.0):
        self.root = root
        self.display_time = int(display_time * 1000)  # in milliseconds
        self.pause_between = int(pause_between * 1000)
        
        # Configure the window to use full-screen mode
        self.root.attributes('-fullscreen', True)
        self.root.config(cursor="none")  # Hide the mouse cursor
        
        # Exit when the ESC key is pressed
        self.root.bind("<Escape>", lambda event: self.root.destroy())
        
        # List of test screens: (Name, Hexadecimal color)
        # IBL neutral background (RGB gray 128,128,128 = #808080)
        self.screens = [
            ("01_IBL_Neutral_Background", "#808080"),
            ("02_Pure_Black", "#000000"),
            ("03_100pct_White", "#FFFFFF"),
            ("04_Pure_Red", "#FF0000"),
            ("05_Pure_Green", "#00FF00"),
            ("06_Pure_Blue", "#0000FF")
        ]
        
        self.current_index = 0
        self.run_sequence()

    def run_sequence(self):
        if self.current_index < len(self.screens):
            name, color = self.screens[self.current_index]
            print(f"Displaying: {name} (Color: {color})")
            
            # Change the window's background color
            self.root.config(bg=color)
            
            # Increment the index and schedule the transition to a black screen
            self.current_index += 1
            self.root.after(self.display_time, self.show_pause)
        else:
            print("=== Calibration Sequence Complete ===")
            self.root.destroy()

    def show_pause(self):
        # Black transition screen
        self.root.config(bg="#000000")
        self.root.after(self.pause_between, self.run_sequence)

if __name__ == "__main__":
    print("=== STARTING CALIBRATION SEQUENCE (TKINTER) ===")
    print("Press 'ESC' at any time to exit.\n")
    
    root = tk.Tk()
    app = CalibrationApp(root, display_time=5.0, pause_between=2.0)
    root.mainloop()

timer_end = time.time()  # Stop the timer
print(f"\n=== Total execution time: {timer_end - timer:.2f} seconds ===")