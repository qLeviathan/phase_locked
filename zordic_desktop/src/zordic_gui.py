"""
ZORDIC Desktop Application - GUI Interface
Leviathan AI Corporation

Real-time visualization of φ-field self-organizing lattice
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from threading import Thread
import time

from zordic_core import PhiLattice, PHI, PSI


class ZordicDesktopApp:
    """Main desktop application for Zordic lattice system"""

    def __init__(self, root):
        self.root = root
        self.root.title("ZORDIC - φ-Field Self-Organizing Lattice")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0d1117')

        # Current lattice
        self.lattice = None
        self.analyzing = False

        # Set dark theme
        self.setup_style()
        self.create_widgets()

    def setup_style(self):
        """Setup dark theme styling"""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure colors
        bg_dark = '#0d1117'
        bg_light = '#161b22'
        fg_color = '#c9d1d9'
        accent = '#58a6ff'

        style.configure('Dark.TFrame', background=bg_dark)
        style.configure('Dark.TLabel', background=bg_dark, foreground=fg_color, font=('Consolas', 10))
        style.configure('Title.TLabel', background=bg_dark, foreground=accent, font=('Consolas', 14, 'bold'))
        style.configure('Dark.TButton', background=bg_light, foreground=fg_color, font=('Consolas', 10))
        style.map('Dark.TButton', background=[('active', accent)])
        style.configure('Dark.TEntry', fieldbackground=bg_light, foreground=fg_color, font=('Consolas', 11))

    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Controls and input
        left_panel = ttk.Frame(main_frame, style='Dark.TFrame', width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Title
        title = ttk.Label(left_panel, text="ZORDIC ENGINE", style='Title.TLabel')
        title.pack(pady=(0, 20))

        # Input section
        input_frame = ttk.LabelFrame(left_panel, text="Input Text", style='Dark.TFrame')
        input_frame.pack(fill=tk.BOTH, padx=10, pady=10)

        self.text_input = tk.Text(input_frame, height=6, width=40, bg='#161b22', fg='#c9d1d9',
                                   font=('Consolas', 11), insertbackground='#58a6ff', relief=tk.FLAT)
        self.text_input.pack(padx=10, pady=10)
        self.text_input.insert('1.0', 'quantum field theory')

        # Temperature control
        temp_frame = ttk.Frame(input_frame, style='Dark.TFrame')
        temp_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Label(temp_frame, text="Temperature:", style='Dark.TLabel').pack(side=tk.LEFT)
        self.temp_var = tk.DoubleVar(value=1.0)
        temp_scale = ttk.Scale(temp_frame, from_=0.1, to=2.0, variable=self.temp_var, orient=tk.HORIZONTAL)
        temp_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.temp_label = ttk.Label(temp_frame, text="1.0", style='Dark.TLabel', width=4)
        self.temp_label.pack(side=tk.LEFT)
        temp_scale.configure(command=lambda v: self.temp_label.config(text=f"{float(v):.1f}"))

        # Control buttons
        btn_frame = ttk.Frame(left_panel, style='Dark.TFrame')
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self.analyze_btn = tk.Button(btn_frame, text="⚡ ANALYZE", command=self.run_analysis,
                                      bg='#21262d', fg='#c9d1d9', font=('Consolas', 11, 'bold'),
                                      relief=tk.FLAT, cursor='hand2', height=2)
        self.analyze_btn.pack(fill=tk.X, pady=(0, 5))

        self.clear_btn = tk.Button(btn_frame, text="🔄 CLEAR", command=self.clear_all,
                                    bg='#21262d', fg='#c9d1d9', font=('Consolas', 10),
                                    relief=tk.FLAT, cursor='hand2')
        self.clear_btn.pack(fill=tk.X)

        # Metrics display
        metrics_frame = ttk.LabelFrame(left_panel, text="System Metrics", style='Dark.TFrame')
        metrics_frame.pack(fill=tk.BOTH, padx=10, pady=10, expand=True)

        self.metrics_text = scrolledtext.ScrolledText(metrics_frame, height=15, bg='#0d1117',
                                                       fg='#58a6ff', font=('Consolas', 9),
                                                       relief=tk.FLAT, state=tk.DISABLED)
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Log display
        log_frame = ttk.LabelFrame(left_panel, text="Operation Log", style='Dark.TFrame')
        log_frame.pack(fill=tk.BOTH, padx=10, pady=10, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, bg='#0d1117',
                                                   fg='#8b949e', font=('Consolas', 8),
                                                   relief=tk.FLAT, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right panel - Visualizations
        right_panel = ttk.Frame(main_frame, style='Dark.TFrame')
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8), facecolor='#0d1117')
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize plots
        self.setup_plots()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               style='Dark.TLabel', relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_plots(self):
        """Setup matplotlib subplots"""
        self.fig.clear()

        # Create 4 subplots
        gs = self.fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        self.ax_phi = self.fig.add_subplot(gs[0, 0])
        self.ax_psi = self.fig.add_subplot(gs[0, 1])
        self.ax_delta = self.fig.add_subplot(gs[1, :])
        self.ax_cascade = self.fig.add_subplot(gs[2, :])

        # Style all axes
        for ax in [self.ax_phi, self.ax_psi, self.ax_delta, self.ax_cascade]:
            ax.set_facecolor('#0d1117')
            ax.tick_params(colors='#8b949e', labelsize=8)
            ax.spines['bottom'].set_color('#30363d')
            ax.spines['top'].set_color('#30363d')
            ax.spines['left'].set_color('#30363d')
            ax.spines['right'].set_color('#30363d')

        self.canvas.draw()

    def update_log(self, message, level='info'):
        """Add message to log"""
        self.log_text.configure(state=tk.NORMAL)

        # Color by level
        tag = level
        if level == 'header':
            self.log_text.insert(tk.END, f"\n{message}\n", tag)
            self.log_text.tag_config(tag, foreground='#58a6ff', font=('Consolas', 8, 'bold'))
        elif level == 'success':
            self.log_text.insert(tk.END, f"{message}\n", tag)
            self.log_text.tag_config(tag, foreground='#3fb950')
        elif level == 'warning':
            self.log_text.insert(tk.END, f"{message}\n", tag)
            self.log_text.tag_config(tag, foreground='#d29922')
        else:
            self.log_text.insert(tk.END, f"{message}\n", tag)
            self.log_text.tag_config(tag, foreground='#8b949e')

        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def update_metrics(self, results):
        """Update metrics display"""
        self.metrics_text.configure(state=tk.NORMAL)
        self.metrics_text.delete('1.0', tk.END)

        metrics_str = f"""
╔══════════════════════════════════╗
║      SYSTEM ANALYSIS RESULTS     ║
╚══════════════════════════════════╝

INITIAL STATE:
  φ-field total:    {results['initial_state']['phi_total']:.3f}
  ψ-field total:    {results['initial_state']['psi_total']:.3f}
  Field delta:      {results['initial_state']['delta_total']:.3f}
  Avg stability:    {results['initial_state']['stability']:.3f}
  Stable nodes:     {results['initial_state']['stable_count']}

CONNECTIVITY:
  Forward (j=1):    {results['connections'].get('j=1', 0)}
  Backward (j=2):   {results['connections'].get('j=2', 0)}
  Bidirect (j=3):   {results['connections'].get('j=3', 0)}
  Disconn (j=0):    {results['connections'].get('j=0', 0)}
  Forbidden:        {results['connections'].get('forbidden', 0)}

FINAL REGIME:
  Type:             {results['regime']['regime']}
  Deterministic:    {results['regime']['ratio']:.1%}
  Stable nodes:     {results['regime']['stable_count']} / {results['regime']['total_nodes']}
  Final stability:  {results['final_stability']:.3f}

PROPERTIES:
  φ = {PHI:.6f}
  ψ = {PSI:.6f}
  φ + ψ = {(PHI + PSI):.6f} ≈ 1
  φ × ψ = {(PHI * PSI):.6f} ≈ -1
"""
        self.metrics_text.insert('1.0', metrics_str)
        self.metrics_text.configure(state=tk.DISABLED)

    def visualize_results(self):
        """Update all visualizations"""
        if not self.lattice or not self.lattice.nodes:
            return

        # Clear plots
        self.ax_phi.clear()
        self.ax_psi.clear()
        self.ax_delta.clear()
        self.ax_cascade.clear()

        positions = list(range(len(self.lattice.nodes)))
        chars = [n.char for n in self.lattice.nodes]
        phi_vals = [n.phi for n in self.lattice.nodes]
        psi_vals = [n.psi for n in self.lattice.nodes]
        delta_vals = [n.delta for n in self.lattice.nodes]

        # φ-field
        self.ax_phi.plot(positions, phi_vals, 'o-', color='#58a6ff', linewidth=2, markersize=6)
        self.ax_phi.set_title('φ-Field (Forward Causal)', color='#c9d1d9', fontsize=10)
        self.ax_phi.set_ylabel('φ amplitude', color='#c9d1d9', fontsize=9)
        self.ax_phi.grid(True, alpha=0.2, color='#30363d')

        # ψ-field
        self.ax_psi.plot(positions, psi_vals, 'o-', color='#f85149', linewidth=2, markersize=6)
        self.ax_psi.set_title('ψ-Field (Backward Causal)', color='#c9d1d9', fontsize=10)
        self.ax_psi.set_ylabel('ψ amplitude', color='#c9d1d9', fontsize=9)
        self.ax_psi.grid(True, alpha=0.2, color='#30363d')

        # Δ-field (stability)
        colors = ['#3fb950' if abs(d) < 0.5 else '#f0883e' for d in delta_vals]
        self.ax_delta.bar(positions, delta_vals, color=colors, alpha=0.7, edgecolor='#30363d')
        self.ax_delta.axhline(y=0.5, color='#3fb950', linestyle='--', alpha=0.5, linewidth=1)
        self.ax_delta.axhline(y=-0.5, color='#3fb950', linestyle='--', alpha=0.5, linewidth=1)
        self.ax_delta.axhline(y=0, color='#8b949e', linestyle='-', alpha=0.3, linewidth=1)
        self.ax_delta.set_title('Field Stability Δ = φ - ψ (green=stable, orange=unstable)',
                                color='#c9d1d9', fontsize=10)
        self.ax_delta.set_ylabel('Δ', color='#c9d1d9', fontsize=9)
        self.ax_delta.set_xlabel('Position', color='#c9d1d9', fontsize=9)
        self.ax_delta.set_xticks(positions)
        self.ax_delta.set_xticklabels(chars, fontsize=8)
        self.ax_delta.grid(True, alpha=0.2, color='#30363d')

        # Cascade convergence
        if self.lattice.cascade_history:
            iterations = [h['iteration'] for h in self.lattice.cascade_history]
            stabilities = [h['avg_stability'] for h in self.lattice.cascade_history]

            self.ax_cascade.plot(iterations, stabilities, 'o-', color='#d29922',
                                linewidth=2, markersize=4)
            self.ax_cascade.axhline(y=0.5, color='#3fb950', linestyle='--',
                                   alpha=0.5, linewidth=1, label='Stability threshold')
            self.ax_cascade.set_title('Cascade Convergence', color='#c9d1d9', fontsize=10)
            self.ax_cascade.set_xlabel('Iteration', color='#c9d1d9', fontsize=9)
            self.ax_cascade.set_ylabel('Avg |Δ|', color='#c9d1d9', fontsize=9)
            self.ax_cascade.legend(loc='upper right', fontsize=8,
                                  facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
            self.ax_cascade.grid(True, alpha=0.2, color='#30363d')

        self.canvas.draw()

    def run_analysis(self):
        """Run the lattice analysis"""
        if self.analyzing:
            return

        text = self.text_input.get('1.0', tk.END).strip()
        if not text:
            messagebox.showwarning("No Input", "Please enter some text to analyze.")
            return

        temperature = self.temp_var.get()

        # Clear previous log
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.configure(state=tk.DISABLED)

        # Update status
        self.status_var.set("Analyzing...")
        self.analyze_btn.config(state=tk.DISABLED, text="⏳ ANALYZING...")
        self.analyzing = True

        # Run analysis in thread
        def analyze():
            try:
                # Create lattice
                self.lattice = PhiLattice(text, temperature)

                # Run full analysis
                results = self.lattice.full_analysis()

                # Update UI in main thread
                self.root.after(0, self._update_after_analysis, results)

            except Exception as e:
                self.root.after(0, self._handle_error, str(e))

        Thread(target=analyze, daemon=True).start()

    def _update_after_analysis(self, results):
        """Update UI after analysis completes"""
        # Update log with all entries
        for entry in self.lattice.log_entries:
            self.update_log(entry['message'], entry['level'])

        # Update metrics
        self.update_metrics(results)

        # Update visualizations
        self.visualize_results()

        # Reset button
        self.analyze_btn.config(state=tk.NORMAL, text="⚡ ANALYZE")
        self.status_var.set(f"Analysis complete - Regime: {results['regime']['regime']}")
        self.analyzing = False

    def _handle_error(self, error_msg):
        """Handle analysis error"""
        self.update_log(f"ERROR: {error_msg}", 'warning')
        self.analyze_btn.config(state=tk.NORMAL, text="⚡ ANALYZE")
        self.status_var.set("Analysis failed")
        self.analyzing = False
        messagebox.showerror("Analysis Error", f"An error occurred:\n{error_msg}")

    def clear_all(self):
        """Clear all displays"""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.configure(state=tk.DISABLED)

        self.metrics_text.configure(state=tk.NORMAL)
        self.metrics_text.delete('1.0', tk.END)
        self.metrics_text.configure(state=tk.DISABLED)

        self.setup_plots()
        self.status_var.set("Ready")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = ZordicDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
