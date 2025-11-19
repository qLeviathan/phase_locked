#!/usr/bin/env python3
"""
Concurrent Model Chat GUI

Desktop application for running multiple AI models side-by-side.
- Integer-Only Phi-Mamba (Zeckendorf-CORDIC)
- Ollama models
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import sys
import os
from typing import List, Dict
import time

# Add paths
sys.path.append('phi_mamba_integer')

try:
    from integer_phi_mamba import IntegerPhiMamba
    HAS_INTEGER_MAMBA = True
except ImportError:
    HAS_INTEGER_MAMBA = False

try:
    import requests
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


class ModelPanel(ttk.Frame):
    """Panel for displaying a single model's output"""

    def __init__(self, parent, model_name: str):
        super().__init__(parent, relief=tk.RIDGE, borderwidth=2)
        self.model_name = model_name

        # Title
        title_frame = ttk.Frame(self)
        title_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(
            title_frame,
            text=model_name,
            font=('Arial', 12, 'bold')
        ).pack(side=tk.LEFT)

        self.status_label = ttk.Label(
            title_frame,
            text="●",
            foreground="green"
        )
        self.status_label.pack(side=tk.RIGHT)

        # Output text area
        self.output_text = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            width=40,
            height=15,
            font=('Consolas', 10)
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Timing label
        self.timing_label = ttk.Label(self, text="", foreground="gray")
        self.timing_label.pack(padx=5, pady=2)

    def set_status(self, status: str, color: str = "green"):
        """Update status indicator"""
        self.status_label.config(text=status, foreground=color)

    def clear_output(self):
        """Clear output text"""
        self.output_text.delete('1.0', tk.END)

    def append_output(self, text: str):
        """Append text to output"""
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)

    def set_timing(self, elapsed: float):
        """Set timing label"""
        self.timing_label.config(text=f"Generated in {elapsed:.2f}s")


class ConcurrentChatGUI:
    """Main GUI application"""

    def __init__(self, root):
        self.root = root
        self.root.title("Concurrent AI Chat - Phi-Mamba & Ollama")
        self.root.geometry("1200x800")

        # Initialize models
        self.models = {}
        self.panels = {}

        self._setup_ui()
        self._load_models()

    def _setup_ui(self):
        """Setup the user interface"""
        # Top frame - Controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)

        ttk.Label(
            control_frame,
            text="Concurrent AI Chat",
            font=('Arial', 16, 'bold')
        ).pack()

        ttk.Label(
            control_frame,
            text="Zeckendorf-CORDIC Integer-Only System + Ollama",
            font=('Arial', 10, 'italic')
        ).pack()

        # Model selection frame
        model_select_frame = ttk.LabelFrame(
            control_frame,
            text="Active Models",
            padding="5"
        )
        model_select_frame.pack(fill=tk.X, pady=10)

        # Checkboxes for models
        self.model_vars = {}
        model_list = [
            ("Phi-Mamba (Integer)", "phi-mamba"),
            ("Ollama: llama3", "llama3"),
            ("Ollama: mistral", "mistral"),
            ("Ollama: codellama", "codellama"),
        ]

        for i, (display_name, model_id) in enumerate(model_list):
            var = tk.BooleanVar(value=(i < 2))  # First 2 enabled by default
            self.model_vars[model_id] = var
            ttk.Checkbutton(
                model_select_frame,
                text=display_name,
                variable=var,
                command=self._update_active_models
            ).grid(row=0, column=i, padx=5)

        # Input frame
        input_frame = ttk.LabelFrame(
            control_frame,
            text="Your Prompt",
            padding="5"
        )
        input_frame.pack(fill=tk.X, pady=5)

        self.prompt_text = scrolledtext.ScrolledText(
            input_frame,
            wrap=tk.WORD,
            height=3,
            font=('Arial', 11)
        )
        self.prompt_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        self.generate_btn = ttk.Button(
            button_frame,
            text="Generate (Concurrent)",
            command=self._generate_concurrent
        )
        self.generate_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Clear All",
            command=self._clear_all
        ).pack(side=tk.LEFT, padx=5)

        # Model panels frame (scrollable)
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.model_canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(
            canvas_frame,
            orient=tk.HORIZONTAL,
            command=self.model_canvas.xview
        )

        self.model_panels_frame = ttk.Frame(self.model_canvas)

        self.model_canvas.create_window(
            (0, 0),
            window=self.model_panels_frame,
            anchor=tk.NW
        )

        self.model_canvas.config(xscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.model_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Status bar
        self.status_bar = ttk.Label(
            self.root,
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _load_models(self):
        """Load available models"""
        self.status_bar.config(text="Loading models...")

        # Load Phi-Mamba
        if HAS_INTEGER_MAMBA:
            try:
                self.models['phi-mamba'] = IntegerPhiMamba(vocab_size=50000)
                self.status_bar.config(text="Phi-Mamba loaded")
            except Exception as e:
                messagebox.showwarning("Model Load Error", f"Could not load Phi-Mamba: {e}")

        self.status_bar.config(text="Ready")
        self._update_active_models()

    def _update_active_models(self):
        """Update displayed model panels based on selection"""
        # Clear existing panels
        for widget in self.model_panels_frame.winfo_children():
            widget.destroy()
        self.panels.clear()

        # Create panels for selected models
        col = 0
        for model_id, var in self.model_vars.items():
            if var.get():
                if model_id == 'phi-mamba':
                    panel = ModelPanel(self.model_panels_frame, "Phi-Mamba Integer")
                else:
                    panel = ModelPanel(self.model_panels_frame, f"Ollama: {model_id}")

                panel.grid(row=0, column=col, padx=5, pady=5, sticky=tk.NSEW)
                self.panels[model_id] = panel
                col += 1

        # Update canvas scroll region
        self.root.update_idletasks()
        self.model_canvas.config(scrollregion=self.model_canvas.bbox('all'))

    def _clear_all(self):
        """Clear all outputs"""
        for panel in self.panels.values():
            panel.clear_output()
            panel.set_timing(0)

    def _generate_concurrent(self):
        """Generate from all selected models concurrently"""
        prompt = self.prompt_text.get('1.0', tk.END).strip()

        if not prompt:
            messagebox.showwarning("Empty Prompt", "Please enter a prompt first")
            return

        if not self.panels:
            messagebox.showwarning("No Models", "Please select at least one model")
            return

        # Clear outputs
        self._clear_all()

        # Disable button
        self.generate_btn.config(state=tk.DISABLED)
        self.status_bar.config(text="Generating...")

        # Update status indicators
        for panel in self.panels.values():
            panel.set_status("⏳", "orange")

        # Run generations in threads
        threads = []
        for model_id, panel in self.panels.items():
            thread = threading.Thread(
                target=self._generate_single,
                args=(model_id, panel, prompt)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)

    def _generate_single(self, model_id: str, panel: ModelPanel, prompt: str):
        """Generate from a single model (runs in thread)"""
        start_time = time.time()

        try:
            if model_id == 'phi-mamba':
                # Integer Phi-Mamba
                if 'phi-mamba' in self.models:
                    result = self.models['phi-mamba'].generate(prompt, max_length=100)
                else:
                    result = "Model not loaded"

            else:
                # Ollama model
                if HAS_OLLAMA:
                    try:
                        response = requests.post(
                            "http://localhost:11434/api/generate",
                            json={
                                "model": model_id,
                                "prompt": prompt,
                                "stream": False,
                                "options": {"num_predict": 100}
                            },
                            timeout=60
                        )

                        if response.status_code == 200:
                            result = response.json().get('response', 'No response')
                        else:
                            result = f"Error: HTTP {response.status_code}"

                    except Exception as e:
                        result = f"Error: {str(e)}\n\nMake sure Ollama is running:\n  ollama serve"
                else:
                    result = "Ollama not available (install: pip install requests)"

            elapsed = time.time() - start_time

            # Update UI (must be done in main thread)
            self.root.after(0, lambda: self._update_panel(panel, result, elapsed, True))

        except Exception as e:
            self.root.after(0, lambda: self._update_panel(panel, f"Error: {str(e)}", 0, False))

    def _update_panel(self, panel: ModelPanel, text: str, elapsed: float, success: bool):
        """Update panel with results (called in main thread)"""
        panel.append_output(text)
        panel.set_timing(elapsed)

        if success:
            panel.set_status("✓", "green")
        else:
            panel.set_status("✗", "red")

        # Check if all done
        all_done = all(
            p.status_label.cget('text') in ["✓", "✗"]
            for p in self.panels.values()
        )

        if all_done:
            self.generate_btn.config(state=tk.NORMAL)
            self.status_bar.config(text="Ready")


def main():
    root = tk.Tk()
    app = ConcurrentChatGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
