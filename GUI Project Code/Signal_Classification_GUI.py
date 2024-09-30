import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.models import load_model
import os

class SignalClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Classifier")
        self.root.geometry("1500x1050")

        # Load the pre-trained model
        model_path = 'C:/Users/bidha/Desktop/Project/CIProject/GUI Project Code/pretrained_model.h5'  # Add your own Preserved model Path Here
        self.model = load_model(model_path)

        # Initialize variables
        self.current_file_data = None
        self.save_directory = None
        self.signal_results = []
        self.current_signal_index = 0

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        self.top_panel = tk.Frame(self.root, padx=10, pady=10)
        self.top_panel.pack(side=tk.TOP, fill=tk.X)

        self.instructions_label = tk.Label(self.top_panel, text="Instructions:", font=("Arial", 12, "bold"))
        self.instructions_label.pack(pady=5)

        self.instructions_text = tk.Label(self.top_panel, text=( 
            "1. Data should be a comma-separated list of numbers.\n" 
            "2. Example: '0.1,0.2,0.3,0.4,0.5'\n" 
            "3. The model supports binary classification (Source #1 or Source #2).\n" 
            "4. Ensure data is properly scaled if necessary.\n" 
            "5. For large datasets, use chunked input and be patient."
        ), justify=tk.LEFT, anchor="w")
        self.instructions_text.pack(pady=5)

        self.signal_input_label = tk.Label(self.top_panel, text="Enter signal data (comma-separated):")
        self.signal_input_label.pack(pady=5)

        self.signal_input = tk.Entry(self.top_panel, width=50)
        self.signal_input.pack(pady=5)

        self.generate_btn = tk.Button(self.top_panel, text="Generate and Classify", command=self.start_processing)
        self.generate_btn.pack(pady=10)

        self.new_btn = tk.Button(self.top_panel, text="New", command=self.new_input)
        self.new_btn.pack(pady=10)

        self.save_btn = tk.Button(self.top_panel, text="Save Results", command=self.save_results)
        self.save_btn.pack(pady=10)

        self.status_label = tk.Label(self.top_panel, text="Status: Ready")
        self.status_label.pack(pady=5)

        self.progress = ttk.Progressbar(self.top_panel, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=5)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.right_panel = tk.Frame(self.root, padx=10, pady=10)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        self.label = tk.Label(self.right_panel, text="Signal Information", font=("Arial", 14))
        self.label.pack(pady=10)

        self.reset_btn = tk.Button(self.top_panel, text="Reset", command=self.reset)
        self.reset_btn.pack(pady=10)

        self.upload_btn = tk.Button(self.top_panel, text="Upload File", command=self.upload_file)
        self.upload_btn.pack(pady=10)

        self.next_btn = tk.Button(self.right_panel, text="Next", command=self.next_signal)
        self.next_btn.pack(pady=5)

        self.previous_btn = tk.Button(self.right_panel, text="Previous", command=self.previous_signal)
        self.previous_btn.pack(pady=5)

        self.predicted_score_label = tk.Label(self.right_panel, text="Predicted Score: N/A", font=("Arial", 12))
        self.predicted_score_label.pack(pady=10)

    def start_processing(self):
        self.update_status("Processing...")
        self.progress.start()
        self.root.after(100, self.generate_and_classify)

    def new_input(self):
        self.signal_input.delete(0, tk.END)
        self.reset()

    def reset(self):
        self.signal_input.delete(0, tk.END)
        self.ax.clear()
        self.canvas.draw()
        self.label.config(text="Signal Information")
        self.predicted_score_label.config(text="Predicted Score: N/A")
        self.current_file_data = None
        self.signal_results = []
        self.current_signal_index = 0
        self.update_status("Ready")

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")])
        if file_path:
            try:
                if file_path.endswith('.xlsx'):
                    self.current_file_data = pd.read_excel(file_path).values
                elif file_path.endswith('.csv'):
                    self.current_file_data = pd.read_csv(file_path).values
                self.update_status(f"Loaded file: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def generate_and_classify(self):
        try:
            signals = []

            if self.signal_input.get():
                # Process input string data
                signal_str = self.signal_input.get().strip()
                signal_list = [float(value) for value in signal_str.replace('\n', ',').split(',')]
                signals.append(signal_list)

            elif self.current_file_data is not None:
                signals = self.current_file_data.tolist()

            else:
                raise ValueError("No signal data entered.")

            # Process each signal
            self.signal_results = []
            for signal in signals:
                label, score = self.classify_signal(signal)
                self.signal_results.append((label, score, signal))

            if self.signal_results:
                self.current_signal_index = 0
                self.display_signal(self.current_signal_index)

            self.update_status("Processing Complete.")
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            print(f"Exception occurred: {str(e)}")
        finally:
            self.progress.stop()

    def classify_signal(self, signal):
        expected_length = 5052
        signal = np.array(signal)

        # Debugging: Print the length of the incoming signal
        print(f"Incoming signal length: {len(signal)}")

        # Check for NaN or Inf values and handle them
        if np.isnan(signal).any() or np.isinf(signal).any():
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure signals are of the expected length
        if signal.shape[0] < expected_length:
            signal = np.pad(signal, (0, expected_length - signal.shape[0]), mode='constant')
            print(f"Padded signal length: {len(signal)}")
        else:
            signal = signal[:expected_length]
            print(f"Trimmed signal length: {len(signal)}")

        # Check for expected input shape
        print(f"Signal shape for model input: {signal.shape}")

        # Reshape the signals to match model input (samples, timesteps, features)
        signal = signal.reshape(1, expected_length, 1)

        # Predict
        predictions = self.model.predict(signal)
        predicted_class = (predictions > 0.5).astype(int).flatten()[0]
        predicted_label = 'Source #1' if predicted_class == 0 else 'Source #2'
        predicted_score = float(predictions[0][0]) * 100

        # Debugging: Output prediction results
        print(f"Predicted Class: {predicted_class}, Label: {predicted_label}, Score: {predicted_score:.2f}")

        return predicted_label, predicted_score

    def display_signal(self, index):
        if 0 <= index < len(self.signal_results):
            predicted_label, predicted_score, signal = self.signal_results[index]

            # Print signal values for debugging
            print(f"Displaying signal: {signal}")

            # Plot signal data
            self.ax.clear()
            self.ax.plot(signal, color='blue')
            self.ax.set_title(f"Signal Plot - Predicted Source: {predicted_label}")
            self.ax.set_xlabel('Time')
            self.ax.set_ylabel('Amplitude')
            self.ax.set_ylim([-1, 1])  # Adjust the y-limits based on expected signal range
            self.ax.grid(True)

            # Update GUI
            self.label.config(text=f"Predicted Source: {predicted_label}")
            self.predicted_score_label.config(text=f"Predicted Score: {predicted_score:.2f}%")
            self.canvas.draw()

    def next_signal(self):
        if self.current_signal_index < len(self.signal_results) - 1:
            self.current_signal_index += 1
            self.display_signal(self.current_signal_index)

    def previous_signal(self):
        if self.current_signal_index > 0:
            self.current_signal_index -= 1
            self.display_signal(self.current_signal_index)

    def save_results(self):
        if self.save_directory is None:
            self.save_directory = filedialog.askdirectory()
        if self.save_directory and self.signal_results:
            current_result = self.signal_results[self.current_signal_index]
            result_text = f"Predicted Source: {current_result[0]}\nPredicted Score: {current_result[1]:.2f}%"
            result_file = os.path.join(self.save_directory, "results.txt")
            with open(result_file, "a") as file:
                file.write(result_text + "\n")
            messagebox.showinfo("Info", f"Results saved to {result_file}")

    def update_status(self, status_text):
        self.status_label.config(text=f"Status: {status_text}")
        self.status_label.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()  
    app = SignalClassifierApp(root)
    root.mainloop()
