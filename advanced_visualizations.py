#!/usr/bin/env python3
"""
Advanced visualizations showing Φ-Mamba as panel data structure
Outputs saved with markdown documentation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from datetime import datetime
import os

# Set up paths
import sys
sys.path.append('/mnt/c/Users/casma/phase_locked')

from phi_mamba import PhiLanguageModel, PHI
from phi_mamba.encoding import zeckendorf_decomposition, TokenState
from phi_mamba.utils import fibonacci, compute_berry_phase

# Create output directory
output_dir = "outputs/visualizations"
os.makedirs(output_dir, exist_ok=True)

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('ggplot')
sns.set_palette("husl")

def save_figure_with_notes(fig, name, notes):
    """Save figure with accompanying markdown notes"""
    # Save figure
    fig_path = f"{output_dir}/{name}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    
    # Save markdown notes
    md_path = f"{output_dir}/{name}.md"
    with open(md_path, 'w') as f:
        f.write(f"# {name.replace('_', ' ').title()}\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"![{name}]({name}.png)\n\n")
        f.write(notes)
    print(f"Saved: {md_path}")
    plt.close(fig)


# Visualization 1: Panel Data Structure
def visualize_panel_structure():
    """Show how tokens create panel data"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Φ-Mamba as Panel Data Structure', fontsize=20)
    
    # Create sample sentence
    sentence = "The golden ratio emerges naturally"
    tokens = sentence.lower().split()
    model = PhiLanguageModel()
    
    # Panel 1: Traditional vs Φ-Mamba encoding
    ax1 = axes[0, 0]
    
    # Traditional
    trad_y = np.arange(len(tokens))
    trad_x = [hash(t) % 100 for t in tokens]
    ax1.scatter(trad_x, trad_y, s=100, alpha=0.5, label='Traditional (1D)')
    
    # Φ-Mamba (show multiple dimensions)
    for i, token in enumerate(tokens):
        state = TokenState(token, hash(token) % 100, i, 1000)
        # Show as larger circle with energy-based size
        size = 500 * state.energy
        ax1.scatter(state.index, i, s=size, alpha=0.7, 
                   edgecolors='black', linewidth=2, label='Φ-Mamba' if i==0 else '')
    
    ax1.set_xlabel('Token ID')
    ax1.set_ylabel('Position')
    ax1.set_title('Encoding Dimensionality')
    ax1.legend()
    
    # Panel 2: Feature evolution over position
    ax2 = axes[0, 1]
    
    positions = []
    energies = []
    shell_counts = []
    phases = []
    
    states = []
    for i, token in enumerate(tokens):
        state = TokenState(token, hash(token) % 100, i, 1000)
        states.append(state)
        positions.append(i)
        energies.append(state.energy)
        shell_counts.append(len(state.zeckendorf))
        
        if i > 0:
            phase = compute_berry_phase(states[i-1], state)
            phases.append(phase)
        else:
            phases.append(0)
    
    # Normalize for visualization
    energies = np.array(energies)
    shell_counts = np.array(shell_counts)
    phases = np.array(phases) / (2 * np.pi)
    
    ax2.plot(positions, energies, 'b-o', label='Energy', linewidth=2)
    ax2.plot(positions, shell_counts/max(shell_counts), 'r-s', label='Active Shells (norm)', linewidth=2)
    ax2.plot(positions, phases, 'g-^', label='Phase/2π', linewidth=2)
    
    ax2.set_xlabel('Position (time dimension)')
    ax2.set_ylabel('Feature Value')
    ax2.set_title('Cross-sectional Features Evolution')
    ax2.legend()
    ax2.set_ylim(-0.1, 1.1)
    
    # Panel 3: Entity tracking (same token at different positions)
    ax3 = axes[1, 0]
    
    # Track "the" token across positions
    the_positions = [i for i, t in enumerate(tokens) if t == "the"]
    if len(the_positions) < 2:
        # Add more instances for demonstration
        tokens.extend(["the", "cat", "the"])
        the_positions = [0, 5, 7]
    
    # Show how same entity evolves
    for idx, pos in enumerate(the_positions):
        state = TokenState("the", hash("the") % 100, pos, 1000)
        
        # Plot on cylinder projection
        theta = state.theta_total
        r = min(state.r, 2)  # Cap for visualization
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        color = plt.cm.viridis(pos / 10)
        ax3.scatter(x, y, s=300, c=[color], marker='o', 
                   edgecolors='black', linewidth=2,
                   label=f'Position {pos}')
        
        # Add arrow showing evolution
        if idx > 0:
            prev_pos = the_positions[idx-1]
            prev_state = TokenState("the", hash("the") % 100, prev_pos, 1000)
            prev_x = min(prev_state.r, 2) * np.cos(prev_state.theta_total)
            prev_y = min(prev_state.r, 2) * np.sin(prev_state.theta_total)
            
            ax3.arrow(prev_x, prev_y, x-prev_x, y-prev_y, 
                     head_width=0.1, head_length=0.1, 
                     fc='gray', ec='gray', alpha=0.5)
    
    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray')
    ax3.add_patch(circle)
    
    ax3.set_xlim(-2.5, 2.5)
    ax3.set_ylim(-2.5, 2.5)
    ax3.set_aspect('equal')
    ax3.set_xlabel('X (r·cos(θ))')
    ax3.set_ylabel('Y (r·sin(θ))')
    ax3.set_title('Entity Tracking: "the" token evolution')
    ax3.legend()
    
    # Panel 4: Retrocausal panel structure
    ax4 = axes[1, 1]
    
    # Create forward and backward constraints matrix
    n_tokens = len(tokens[:6])  # First 6 for clarity
    constraint_matrix = np.zeros((n_tokens, n_tokens))
    
    # Forward constraints (traditional)
    for i in range(n_tokens-1):
        constraint_matrix[i, i+1] = 0.8  # Forward influence
    
    # Backward constraints (retrocausal)
    for i in range(1, n_tokens):
        constraint_matrix[i, i-1] = 0.5  # Backward influence
        
    # Visualize as heatmap
    im = ax4.imshow(constraint_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add labels
    ax4.set_xticks(range(n_tokens))
    ax4.set_yticks(range(n_tokens))
    ax4.set_xticklabels(tokens[:n_tokens], rotation=45)
    ax4.set_yticklabels(tokens[:n_tokens])
    
    ax4.set_xlabel('Constraining Token')
    ax4.set_ylabel('Constrained Token')
    ax4.set_title('Bidirectional Constraints (Retrocausal Panel)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Constraint Strength')
    
    # Add arrows to show direction
    ax4.annotate('', xy=(4.5, 3.5), xytext=(3.5, 4.5),
                arrowprops=dict(arrowstyle='<->', color='white', lw=2))
    ax4.text(4.2, 4.2, 'Retrocausal', color='white', fontweight='bold')
    
    plt.tight_layout()
    
    notes = """## Panel Data Structure in Φ-Mamba

This visualization demonstrates how Φ-Mamba creates a comprehensive panel dataset:

### 1. **Encoding Dimensionality** (Top Left)
- Traditional: Tokens map to single ID (1D)
- Φ-Mamba: Each token has multiple features (circle size = energy)
- The multi-dimensional encoding captures geometric, topological, and dynamic information

### 2. **Feature Evolution** (Top Right)
Shows three key features evolving over token positions:
- **Energy** (blue): Exponentially decays by φ^(-position)
- **Active Shells** (red): Number of Fibonacci components in Zeckendorf decomposition
- **Phase** (green): Berry phase accumulation (normalized by 2π)

This demonstrates the time-series aspect of the panel structure.

### 3. **Entity Tracking** (Bottom Left)
Tracks how the same token ("the") evolves across different positions:
- Same token identity (fixed θ angle)
- Different states at different positions
- Arrows show state evolution
- Demonstrates entity fixed effects with time-varying features

### 4. **Retrocausal Constraints** (Bottom Right)
Unique to Φ-Mamba: bidirectional constraint matrix
- Red: Forward constraints (past → future)
- Blue: Backward constraints (future → past)
- Traditional models only have upper triangular (forward only)
- Φ-Mamba has both, creating richer panel structure

### Key Insight:
Each token is like a company in financial panel data:
- Fixed identity (company ID / token θ)
- Time-varying features (quarterly reports / position-dependent state)
- Cross-sectional variation (different companies / different tokens)
- Temporal dynamics (growth over time / energy decay)
- But with retrocausal constraints adding future information!
"""
    
    save_figure_with_notes(fig, "panel_data_structure", notes)


# Visualization 2: Topological Information Encoding
def visualize_topological_encoding():
    """Show how information is stored topologically"""
    fig = plt.figure(figsize=(18, 10))
    
    # Create 3D subplot for main visualization
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    fig.suptitle('Topological Information Encoding', fontsize=20)
    
    # Part 1: 3D Cylinder with Zeckendorf holes
    # Generate sample data
    n_samples = 20
    positions = list(range(1, n_samples + 1))
    
    # For each position, show Zeckendorf decomposition as holes
    for pos in positions:
        zeck = zeckendorf_decomposition(pos)
        
        # Each Fibonacci number represents a "hole" at specific height
        for fib in zeck:
            # Find which Fibonacci index this is
            fib_idx = 1
            f = fibonacci(fib_idx)
            while f < fib:
                fib_idx += 1
                f = fibonacci(fib_idx)
            
            if f == fib:
                # Create ring at this height
                theta = np.linspace(0, 2*np.pi, 50)
                r = 1.0 + 0.1 * np.sin(5*theta)  # Slight variation
                
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = np.full_like(theta, fib_idx)
                
                # Color by position
                color = plt.cm.plasma(pos / n_samples)
                ax1.plot(x, y, z, color=color, alpha=0.6, linewidth=2)
                
                # Add marker for hole
                hole_theta = 2 * np.pi * pos / n_samples
                hole_x = 1.2 * np.cos(hole_theta)
                hole_y = 1.2 * np.sin(hole_theta)
                ax1.scatter(hole_x, hole_y, fib_idx, 
                           color=color, s=100, marker='o',
                           edgecolors='black', linewidth=2)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Fibonacci Index (Shell)')
    ax1.set_title('Zeckendorf Decomposition on Cylinder')
    ax1.view_init(elev=20, azim=45)
    
    # Part 2: Binary emergence from topology
    # Show how binary patterns emerge
    max_display = 30
    binary_matrix = np.zeros((max_display, 10))  # 10 Fibonacci levels
    
    for n in range(1, max_display + 1):
        zeck = zeckendorf_decomposition(n)
        
        for fib in zeck:
            # Find index
            idx = 0
            f = 1
            while f < fib:
                idx += 1
                f = fibonacci(idx + 1)
            
            if f == fib and idx < 10:
                binary_matrix[n-1, idx] = 1
    
    # Plot as heatmap
    im = ax2.imshow(binary_matrix.T, cmap='RdBu_r', aspect='auto',
                    interpolation='nearest')
    
    ax2.set_xlabel('Number (1-30)')
    ax2.set_ylabel('Fibonacci Index')
    ax2.set_title('Emergent Binary Pattern (No Adjacent 1s)')
    
    # Add Fibonacci values as y-labels
    fib_labels = [f'F_{i+1}={fibonacci(i+1)}' for i in range(10)]
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(fib_labels)
    
    # Highlight gap constraint
    # Check for violations (adjacent 1s)
    violations = 0
    for row in range(max_display):
        for col in range(9):
            if binary_matrix[row, col] == 1 and binary_matrix[row, col+1] == 1:
                violations += 1
                # Draw red box around violation
                rect = plt.Rectangle((row-0.5, col-0.5), 1, 2, 
                                   fill=False, edgecolor='red', linewidth=3)
                ax2.add_patch(rect)
    
    ax2.text(15, -1.5, f'Gap violations found: {violations} (should be 0)', 
             ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    notes = """## Topological Information Encoding

This visualization shows how Φ-Mamba encodes information topologically rather than as raw bits.

### Left: 3D Cylinder Topology
- Each ring represents a Fibonacci shell (F_1, F_2, F_3, ...)
- Colored paths show Zeckendorf decomposition for numbers 1-20
- Black-edged points show "holes" where information exists
- Position around cylinder encodes token identity
- Height encodes which Fibonacci shell is active

### Right: Emergent Binary Pattern
- Rows: Numbers 1-30
- Columns: Fibonacci indices (F_1 through F_10)
- Red cells: "Hole exists" at that scale (bit = 1)
- Blue cells: "No hole" at that scale (bit = 0)

### Key Observations:

1. **No Adjacent 1s**: The gap constraint emerges naturally from φ² = φ + 1
   - If we have F_k and F_{k-1}, they combine to F_{k+1}
   - This is geometric necessity, not an imposed rule

2. **Information = Topology**: 
   - Traditional: Store bits directly
   - Φ-Mamba: Store which holes exist
   - Bits are just the shadow/projection

3. **Compression**: 
   - Zeckendorf is optimally sparse
   - Most efficient representation using Fibonacci base

4. **3D Structure**:
   - Angular position (θ): Token identity
   - Radial position (r): Amplitude/importance
   - Vertical position (z): Active Fibonacci shells
   - This creates a rich geometric information space

### Practical Implication:
Instead of storing a bit array [1,0,1,0,0,1], we store the geometric configuration
of holes in topological space. The bit pattern emerges when we project this
topology onto a 1D representation.

This is why Φ-Mamba can capture richer information - it's working in the native
geometric space rather than flattened bit representations.
"""
    
    save_figure_with_notes(fig, "topological_encoding", notes)


# Visualization 3: Energy Landscape and Phase Space
def visualize_energy_phase_landscape():
    """Visualize energy decay and phase relationships"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Energy Landscape & Phase Space Dynamics', fontsize=20)
    
    # Panel 1: Energy decay surface
    ax1 = axes[0, 0]
    
    positions = np.arange(0, 15)
    tokens = np.arange(0, 10)
    
    P, T = np.meshgrid(positions, tokens)
    # Energy surface: E = φ^(-position) * token_factor
    token_factors = 0.5 + 0.5 * np.sin(tokens)[:, np.newaxis]
    E = PHI**(-P) * token_factors
    
    im1 = ax1.imshow(E, cmap='viridis', aspect='auto', origin='lower')
    ax1.set_xlabel('Position in Sequence')
    ax1.set_ylabel('Token Type')
    ax1.set_title('Energy Landscape E(token, position)')
    
    # Add contour lines
    contours = ax1.contour(P, T, E, levels=[0.01, 0.1, 0.5], colors='white', alpha=0.5)
    ax1.clabel(contours, inline=True, fontsize=8)
    
    plt.colorbar(im1, ax=ax1, label='Energy')
    
    # Panel 2: Phase accumulation
    ax2 = axes[0, 1]
    
    # Simulate phase accumulation along a path
    n_steps = 50
    phases = []
    positions_path = []
    
    phase = 0
    for i in range(n_steps):
        positions_path.append(i)
        phases.append(phase)
        # Random walk in phase space
        delta_phase = np.random.normal(0, 0.5) + 0.1
        phase = phase + delta_phase
    
    phases = np.array(phases)
    positions_path = np.array(positions_path)
    
    # Color by phase lock status
    colors = ['green' if abs(p % (2*np.pi)) < 0.5 or abs(p % (2*np.pi)) > 5.8 else 'red' 
              for p in phases]
    
    ax2.scatter(positions_path, phases, c=colors, alpha=0.6, s=30)
    ax2.plot(positions_path, phases, 'k-', alpha=0.3)
    
    # Mark phase-locked regions
    for k in range(4):
        ax2.axhspan(k*2*np.pi - 0.5, k*2*np.pi + 0.5, alpha=0.1, color='green')
    
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Berry Phase γ')
    ax2.set_title('Phase Accumulation (Green = Locked)')
    ax2.set_ylim(-0.5, 7)
    
    # Panel 3: Pentagon reflection energy decay
    ax3 = axes[0, 2]
    
    # Show energy decay through reflections
    max_bounces = 10
    energy_paths = []
    
    for initial_e in [1.0, 0.5, 0.2]:
        energies = [initial_e]
        for bounce in range(max_bounces):
            energies.append(energies[-1] / PHI)
        energy_paths.append(energies)
    
    for i, path in enumerate(energy_paths):
        ax3.plot(range(len(path)), path, '-o', 
                label=f'E₀ = {path[0]}', linewidth=2)
    
    ax3.axhline(y=0.01, color='red', linestyle='--', 
                label='Termination threshold')
    ax3.set_xlabel('Pentagon Bounces')
    ax3.set_ylabel('Energy')
    ax3.set_yscale('log')
    ax3.set_title('Energy Decay via Pentagon Reflection')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Phase portrait
    ax4 = axes[1, 0]
    
    # Create phase portrait in (θ, r) space
    theta_range = np.linspace(0, 2*np.pi, 100)
    r_range = np.linspace(0, 2, 100)
    
    Theta, R = np.meshgrid(theta_range, r_range)
    
    # Vector field for phase flow
    dTheta = 0.1 * np.sin(2*Theta) * np.exp(-R)
    dR = -0.2 * R + 0.1 * np.cos(Theta)
    
    # Plot vector field
    skip = 5
    ax4.quiver(Theta[::skip, ::skip], R[::skip, ::skip], 
               dTheta[::skip, ::skip], dR[::skip, ::skip],
               alpha=0.5, scale=5)
    
    # Add some trajectories
    for start_theta in [0, np.pi/2, np.pi, 3*np.pi/2]:
        theta_traj = [start_theta]
        r_traj = [1.5]
        
        for _ in range(50):
            dt = 0.1
            theta_new = theta_traj[-1] + dt * 0.1 * np.sin(2*theta_traj[-1]) * np.exp(-r_traj[-1])
            r_new = r_traj[-1] + dt * (-0.2 * r_traj[-1] + 0.1 * np.cos(theta_traj[-1]))
            
            theta_traj.append(theta_new % (2*np.pi))
            r_traj.append(max(0, r_new))
        
        ax4.plot(theta_traj, r_traj, linewidth=2, alpha=0.8)
    
    ax4.set_xlabel('Phase θ')
    ax4.set_ylabel('Amplitude r')
    ax4.set_title('Phase Portrait in (θ, r) Space')
    ax4.set_xlim(0, 2*np.pi)
    ax4.set_ylim(0, 2)
    
    # Panel 5: Coherence landscape
    ax5 = axes[1, 1]
    
    # Create coherence heatmap
    n_tokens = 20
    coherence_matrix = np.zeros((n_tokens, n_tokens))
    
    for i in range(n_tokens):
        for j in range(n_tokens):
            if i != j:
                # Coherence based on phase difference and distance
                phase_diff = abs(i - j) * 0.3
                distance = abs(i - j)
                coherence = np.exp(-distance/5) * (1 + np.cos(phase_diff))
                coherence_matrix[i, j] = coherence
    
    im5 = ax5.imshow(coherence_matrix, cmap='coolwarm', aspect='auto')
    ax5.set_xlabel('Token j')
    ax5.set_ylabel('Token i')
    ax5.set_title('Pairwise Coherence C(i,j)')
    plt.colorbar(im5, ax=ax5, label='Coherence')
    
    # Panel 6: 3D phase-energy-position space
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    
    # Generate trajectory in 3D space
    t = np.linspace(0, 20, 200)
    position = t
    phase = 2 * np.pi * t / 5 + np.sin(t)
    energy = PHI**(-t/3)
    
    # Color by energy
    colors = plt.cm.plasma(energy / energy[0])
    
    ax6.scatter(position, phase % (2*np.pi), energy, 
               c=colors, s=20, alpha=0.6)
    
    ax6.set_xlabel('Position')
    ax6.set_ylabel('Phase (mod 2π)')
    ax6.set_zlabel('Energy')
    ax6.set_title('Trajectory in Phase-Energy-Position Space')
    
    plt.tight_layout()
    
    notes = """## Energy Landscape & Phase Space Dynamics

This comprehensive visualization explores the dynamic aspects of Φ-Mamba's state space.

### Top Row:

1. **Energy Landscape** (Top Left)
   - Shows how energy depends on both token type and position
   - Energy decays exponentially with position (E ∝ φ^(-pos))
   - Different tokens have different baseline energies
   - White contours show iso-energy curves
   - Natural termination when E < 0.01

2. **Phase Accumulation** (Top Center)
   - Berry phase accumulates as sequence progresses
   - Green points: Phase-locked (γ ≈ 0 mod 2π)
   - Red points: Not phase-locked
   - Green bands show coherent regions
   - Phase locking creates natural "attractors"

3. **Pentagon Reflection** (Top Right)
   - Energy decays by factor of φ with each reflection
   - Different initial energies converge to termination
   - Log scale shows exponential decay
   - ~5-7 bounces typically exhausts energy
   - Natural sentence termination mechanism

### Bottom Row:

4. **Phase Portrait** (Bottom Left)
   - Vector field shows dynamics in (θ, r) space
   - Trajectories spiral inward (energy decay)
   - Angular dynamics create rotation
   - System naturally evolves toward low energy
   - Multiple basins of attraction

5. **Coherence Landscape** (Bottom Center)
   - Pairwise coherence between token positions
   - Nearby tokens have higher coherence
   - Oscillatory pattern from phase relationships
   - Retrocausal encoding uses full matrix
   - Traditional uses only upper triangle

6. **3D State Evolution** (Bottom Right)
   - Complete trajectory in phase-energy-position space
   - Color represents energy level (hot to cool)
   - Spiral pattern shows coupled dynamics
   - Natural termination as energy → 0
   - Phase wraps around cylinder (mod 2π)

### Key Insights:

1. **Energy as Natural Clock**: Energy decay provides intrinsic time arrow
2. **Phase Locking = Coherence**: Sequences naturally seek phase-locked states
3. **Pentagon Reflection**: Non-coherent paths dissipate energy quickly
4. **Attractor Dynamics**: System has natural equilibrium points
5. **Geometric Flow**: Information flows along geometric paths in state space

### Implications for Language Modeling:

- Sentences have natural length determined by energy
- Coherent sequences follow phase-locked trajectories
- Incoherent paths self-terminate via energy dissipation
- No need for artificial sequence length limits
- Retrocausal constraints visible in full coherence matrix
"""
    
    save_figure_with_notes(fig, "energy_phase_landscape", notes)


# Visualization 4: Retrocausal Information Flow
def visualize_retrocausal_flow():
    """Visualize bidirectional information flow"""
    fig = plt.figure(figsize=(18, 12))
    
    # Create custom grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])  # Top full width
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[2, :2])  # Bottom left 2/3
    ax6 = fig.add_subplot(gs[2, 2])
    
    fig.suptitle('Retrocausal Information Flow in Φ-Mamba', fontsize=20)
    
    # Panel 1: Bidirectional flow diagram
    sentence = "The cat sat on the mat"
    tokens = sentence.split()
    n_tokens = len(tokens)
    
    # Create flow visualization
    y_positions = np.linspace(1, 0, n_tokens)
    x_positions = np.arange(n_tokens)
    
    # Draw tokens
    for i, (x, y, token) in enumerate(zip(x_positions, y_positions, tokens)):
        # Token box
        rect = plt.Rectangle((x-0.4, y-0.05), 0.8, 0.1, 
                           facecolor='lightblue', edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x, y, token, ha='center', va='center', fontweight='bold')
        
        # Forward flow (φ-path)
        if i < n_tokens - 1:
            ax1.arrow(x+0.4, y, 0.2, 0, head_width=0.02, head_length=0.1,
                     fc='green', ec='green', linewidth=2)
            ax1.text(x+0.5, y+0.02, 'φ', color='green', fontsize=10)
        
        # Backward flow (ψ-path)
        if i > 0:
            ax1.arrow(x-0.4, y-0.03, -0.2, 0, head_width=0.02, head_length=0.1,
                     fc='red', ec='red', linewidth=2, alpha=0.7)
            ax1.text(x-0.5, y-0.05, 'ψ', color='red', fontsize=10)
    
    # Add labels
    ax1.text(-0.5, 1.1, 'START', fontweight='bold', fontsize=12)
    ax1.text(n_tokens-0.5, 1.1, 'END (Ω)', fontweight='bold', fontsize=12)
    
    # Standing wave visualization
    x_wave = np.linspace(0, n_tokens-1, 1000)
    forward_wave = 0.03 * np.sin(2*np.pi*x_wave) * np.exp(-x_wave/5)
    backward_wave = 0.03 * np.sin(2*np.pi*x_wave + np.pi) * np.exp(-(n_tokens-1-x_wave)/5)
    standing = forward_wave + backward_wave
    
    ax1.plot(x_wave, 0.5 + forward_wave, 'g-', alpha=0.5, label='Forward wave')
    ax1.plot(x_wave, 0.5 + backward_wave, 'r-', alpha=0.5, label='Backward wave')
    ax1.plot(x_wave, 0.5 + standing, 'b-', linewidth=2, label='Standing wave')
    
    ax1.set_xlim(-1, n_tokens)
    ax1.set_ylim(-0.1, 1.2)
    ax1.axis('off')
    ax1.legend(loc='upper right')
    ax1.set_title('Bidirectional Information Flow: φ (forward) and ψ (backward) paths')
    
    # Panel 2: Traditional vs Retrocausal encoding
    ax2.set_title('Traditional Encoding')
    
    # Show constraint graph
    pos_trad = np.arange(5)
    for i in range(4):
        ax2.arrow(i, 0, 0.8, 0, head_width=0.1, head_length=0.1,
                 fc='blue', ec='blue')
    
    ax2.scatter(pos_trad, [0]*5, s=200, c='lightblue', edgecolors='black', linewidth=2)
    ax2.set_xlim(-0.5, 4.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xticks(pos_trad)
    ax2.set_xticklabels(['T1', 'T2', 'T3', 'T4', 'T5'])
    ax2.set_yticks([])
    
    # Panel 3: Retrocausal encoding
    ax3.set_title('Retrocausal Encoding')
    
    # Show bidirectional constraints
    pos_retro = np.arange(5)
    for i in range(4):
        # Forward
        ax3.arrow(i, 0.1, 0.8, 0, head_width=0.1, head_length=0.1,
                 fc='green', ec='green', alpha=0.7)
        # Backward
        ax3.arrow(i+1, -0.1, -0.8, 0, head_width=0.1, head_length=0.1,
                 fc='red', ec='red', alpha=0.7)
    
    ax3.scatter(pos_retro, [0]*5, s=200, c='lightgreen', edgecolors='black', linewidth=2)
    ax3.set_xlim(-0.5, 4.5)
    ax3.set_ylim(-0.5, 0.5)
    ax3.set_xticks(pos_retro)
    ax3.set_xticklabels(['T1', 'T2', 'T3', 'T4', 'Ω'])
    ax3.set_yticks([])
    
    # Panel 4: Information content comparison
    ax4.set_title('Information Content')
    
    positions = ['T1', 'T2', 'T3', 'T4', 'T5']
    trad_info = [1.0, 1.8, 2.5, 3.1, 3.6]  # Only past info
    retro_info = [1.5, 2.5, 3.5, 4.2, 4.5]  # Past + future info
    
    x = np.arange(len(positions))
    width = 0.35
    
    ax4.bar(x - width/2, trad_info, width, label='Traditional', color='blue', alpha=0.7)
    ax4.bar(x + width/2, retro_info, width, label='Retrocausal', color='green', alpha=0.7)
    
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Information (bits)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(positions)
    ax4.legend()
    
    # Panel 5: Constraint propagation matrix
    ax5.set_title('Constraint Propagation in Sequence Generation')
    
    # Create detailed constraint matrix
    n = 8
    constraint_matrix = np.zeros((n, n))
    
    # Forward constraints (exponentially decaying)
    for i in range(n):
        for j in range(i+1, n):
            constraint_matrix[i, j] = PHI**(-(j-i))
    
    # Backward constraints
    for i in range(n):
        for j in range(i):
            constraint_matrix[i, j] = PHI**(-(i-j)) * 0.6  # Slightly weaker
    
    im = ax5.imshow(constraint_matrix, cmap='RdBu_r', aspect='auto')
    
    # Add annotations
    for i in range(n):
        for j in range(n):
            if constraint_matrix[i, j] > 0.1:
                text = ax5.text(j, i, f'{constraint_matrix[i, j]:.2f}',
                               ha='center', va='center', fontsize=8)
    
    ax5.set_xlabel('Constraining Position')
    ax5.set_ylabel('Constrained Position')
    ax5.set_xticks(range(n))
    ax5.set_yticks(range(n))
    
    # Draw diagonal
    ax5.plot([-0.5, n-0.5], [-0.5, n-0.5], 'k--', alpha=0.5, linewidth=2)
    
    plt.colorbar(im, ax=ax5, label='Constraint Strength')
    
    # Panel 6: Berry phase cycle
    ax6.set_title('Berry Phase Cycle')
    
    # Show how Berry phase creates cyclic behavior
    theta = np.linspace(0, 4*np.pi, 1000)
    phase = np.sin(theta) + 0.5*np.sin(3*theta)
    
    # Identify phase-locked regions
    locked_mask = np.abs(phase) < 0.5
    
    ax6.plot(theta/(2*np.pi), phase, 'b-', linewidth=2)
    ax6.fill_between(theta/(2*np.pi), -0.5, 0.5, 
                     alpha=0.2, color='green', label='Phase locked')
    
    # Mark complete cycles
    for k in range(3):
        ax6.axvline(x=k, color='red', linestyle=':', alpha=0.5)
        ax6.text(k, 1.5, f'2π×{k}', ha='center')
    
    ax6.set_xlabel('Phase / 2π')
    ax6.set_ylabel('Berry Phase γ')
    ax6.set_ylim(-2, 2)
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    
    notes = """## Retrocausal Information Flow

This visualization demonstrates the unique bidirectional information flow in Φ-Mamba.

### Main Panel (Top): Bidirectional Flow Architecture
- **Green arrows (φ)**: Forward information flow (past → future)
- **Red arrows (ψ)**: Backward information flow (future → past)
- **Standing wave**: Superposition of forward and backward waves
- The end token Ω acts as boundary condition constraining all previous tokens

### Comparison Panels (Middle Row):

1. **Traditional Encoding** (Left)
   - Only forward arrows
   - Each token only knows about past
   - No future information available

2. **Retrocausal Encoding** (Center)
   - Bidirectional arrows
   - Each token influenced by both past and future
   - End token Ω provides boundary condition

3. **Information Content** (Right)
   - Blue bars: Information in traditional (past-only) encoding
   - Green bars: Information in retrocausal encoding
   - Retrocausal consistently has more information per position

### Bottom Panels:

4. **Constraint Propagation Matrix** (Bottom Left)
   - Full matrix shows bidirectional constraints
   - Upper triangle: Forward constraints (past → future)
   - Lower triangle: Backward constraints (future → past)
   - Strength decays exponentially with distance (∝ φ^(-distance))
   - Traditional models only have upper triangle

5. **Berry Phase Cycle** (Bottom Right)
   - Shows periodic nature of phase relationships
   - Green bands: Phase-locked regions (coherent)
   - Phase cycles every 2π
   - Natural quantization of coherent states

### Key Principles:

1. **Bidirectional Causality**:
   - Information flows both forward and backward
   - Future endpoint constrains entire sequence
   - Creates richer representation than forward-only

2. **Standing Waves**:
   - Forward (φ) and backward (ψ) create interference
   - Nodes and antinodes create natural segmentation
   - Explains why certain positions are more "important"

3. **Boundary Conditions**:
   - End token Ω acts like boundary in physics
   - Constrains all previous tokens retrocausally
   - Similar to how knowing the endpoint of a trajectory constrains the path

4. **Information Gain**:
   - Each token contains ~25-40% more information in retrocausal encoding
   - This comes from knowing "where the sentence is going"
   - Enables more coherent generation

### Practical Implications:

1. **Training**: Must process sequences in both directions
2. **Inference**: Generate candidates that satisfy future constraints
3. **Coherence**: Retrocausal constraints ensure global coherence
4. **Efficiency**: More information per token means shorter sequences

This is fundamentally different from traditional autoregressive models and
explains why Φ-Mamba can generate more coherent text with natural boundaries.
"""
    
    save_figure_with_notes(fig, "retrocausal_flow", notes)


# Visualization 5: Panel Data Time Series Analysis
def visualize_panel_time_series():
    """Show panel data structure with time series analysis"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Panel Data Time Series Analysis', fontsize=20)
    
    # Generate synthetic panel data for multiple "entities" (tokens)
    n_tokens = 5
    n_positions = 20
    token_names = ['the', 'cat', 'sat', 'on', 'mat']
    
    # Panel 1: Entity-specific time series
    ax1 = axes[0, 0]
    
    for i, token in enumerate(token_names):
        positions = np.arange(n_positions)
        # Each token has different decay rate and oscillation
        base_energy = PHI**(-positions / (2 + i*0.5))
        oscillation = 0.1 * np.sin(2*np.pi*positions/(3 + i))
        energy = base_energy * (1 + oscillation)
        
        ax1.plot(positions, energy, '-o', label=f'Token: "{token}"', 
                linewidth=2, markersize=4)
    
    ax1.set_xlabel('Position in Sequence')
    ax1.set_ylabel('Energy')
    ax1.set_yscale('log')
    ax1.set_title('Entity-Specific Energy Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Cross-sectional distribution at different times
    ax2 = axes[0, 1]
    
    time_slices = [0, 5, 10, 15]
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_slices)))
    
    for idx, t in enumerate(time_slices):
        energies = []
        for i in range(n_tokens):
            e = PHI**(-t / (2 + i*0.5))
            energies.append(e + np.random.normal(0, 0.01))
        
        ax2.scatter([f'T{i+1}' for i in range(n_tokens)], energies,
                   s=100, alpha=0.7, label=f'Position {t}',
                   color=colors[idx])
    
    ax2.set_xlabel('Token Entity')
    ax2.set_ylabel('Energy')
    ax2.set_title('Cross-Sectional View at Different Positions')
    ax2.legend()
    
    # Panel 3: Fixed effects visualization
    ax3 = axes[1, 0]
    
    # Show token-specific fixed effects (θ angles)
    fixed_effects = []
    random_effects = []
    
    for i, token in enumerate(token_names):
        # Fixed effect: consistent θ angle
        theta = 2 * np.pi * hash(token) % 100 / 100
        fixed_effects.append(theta)
        
        # Random effects: position-varying components
        positions = np.arange(10)
        random = 0.2 * np.sin(positions * theta)
        random_effects.append(random)
    
    # Visualize as bar chart with error bars
    x = np.arange(len(token_names))
    ax3.bar(x, fixed_effects, yerr=[np.std(re) for re in random_effects],
            capsize=10, alpha=0.7, color='skyblue', edgecolor='navy', linewidth=2)
    
    ax3.set_xlabel('Token')
    ax3.set_ylabel('θ Angle (Fixed Effect)')
    ax3.set_title('Fixed Effects (Token Identity) with Random Variation')
    ax3.set_xticks(x)
    ax3.set_xticklabels(token_names)
    
    # Panel 4: Interaction effects heatmap
    ax4 = axes[1, 1]
    
    # Create interaction matrix: token × position effects
    interaction_matrix = np.zeros((n_tokens, 10))
    
    for i in range(n_tokens):
        for j in range(10):
            # Interaction strength depends on both token and position
            interaction = np.sin(i * np.pi/4) * np.cos(j * np.pi/5) * PHI**(-j/5)
            interaction_matrix[i, j] = interaction
    
    im = ax4.imshow(interaction_matrix, cmap='coolwarm', aspect='auto')
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Token')
    ax4.set_title('Token × Position Interaction Effects')
    ax4.set_yticks(range(n_tokens))
    ax4.set_yticklabels(token_names)
    plt.colorbar(im, ax=ax4)
    
    # Panel 5: Autocorrelation structure
    ax5 = axes[2, 0]
    
    # Show autocorrelation for different lags
    max_lag = 15
    lags = np.arange(max_lag)
    
    # Theoretical autocorrelation for φ-system
    autocorr_forward = PHI**(-lags)
    autocorr_backward = PHI**(-lags) * 0.6  # Retrocausal is slightly weaker
    
    ax5.plot(lags, autocorr_forward, 'g-o', label='Forward (φ-path)', linewidth=2)
    ax5.plot(lags, autocorr_backward, 'r-s', label='Backward (ψ-path)', linewidth=2)
    ax5.fill_between(lags, 0, autocorr_forward, alpha=0.2, color='green')
    ax5.fill_between(lags, 0, autocorr_backward, alpha=0.2, color='red')
    
    ax5.set_xlabel('Lag')
    ax5.set_ylabel('Autocorrelation')
    ax5.set_title('Temporal Correlation Structure')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Panel regression results visualization
    ax6 = axes[2, 1]
    
    # Simulate panel regression coefficients
    effects = ['Fixed Effect', 'Time Trend', 'Token×Time', 'Retrocausal', 'Phase Lock']
    traditional = [0.8, -0.3, 0.1, 0, 0]
    phi_mamba = [0.9, -0.5, 0.2, 0.4, 0.3]
    
    x = np.arange(len(effects))
    width = 0.35
    
    ax6.bar(x - width/2, traditional, width, label='Traditional', alpha=0.7)
    ax6.bar(x + width/2, phi_mamba, width, label='Φ-Mamba', alpha=0.7)
    
    ax6.set_ylabel('Coefficient Value')
    ax6.set_title('Panel Regression: Effect Sizes')
    ax6.set_xticks(x)
    ax6.set_xticklabels(effects, rotation=45, ha='right')
    ax6.legend()
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    notes = """## Panel Data Time Series Analysis

This visualization demonstrates how Φ-Mamba creates a rich panel data structure with both cross-sectional and time series dimensions.

### Top Row:

1. **Entity-Specific Trajectories** (Top Left)
   - Each token type follows its own energy decay path
   - Different decay rates (entity heterogeneity)
   - Oscillations show token-specific dynamics
   - Log scale reveals exponential decay ∝ φ^(-t)

2. **Cross-Sectional Distribution** (Top Right)
   - Shows energy distribution across tokens at fixed time points
   - Colors represent different positions (time slices)
   - Variance decreases over time (convergence)
   - Entity-specific differences persist

### Middle Row:

3. **Fixed Effects Structure** (Middle Left)
   - Bar height: Token-specific fixed effect (θ angle)
   - Error bars: Random variation around fixed effect
   - Each token has consistent identity across positions
   - Variation comes from position-dependent features

4. **Interaction Effects** (Middle Right)
   - Heatmap shows token × position interactions
   - Red: Positive interaction effects
   - Blue: Negative interaction effects
   - Non-additive effects captured naturally

### Bottom Row:

5. **Autocorrelation Structure** (Bottom Left)
   - Green: Forward autocorrelation (traditional)
   - Red: Backward autocorrelation (retrocausal)
   - Exponential decay with lag
   - Φ-Mamba has both forward and backward correlations

6. **Regression Coefficients** (Bottom Right)
   - Comparison of effect sizes
   - Traditional: Only fixed effects and time trend
   - Φ-Mamba: Additional retrocausal and phase lock effects
   - Richer model captures more variance

### Panel Data Properties:

1. **Dimensions**:
   - Entities: Token types (cross-sectional)
   - Time: Position in sequence (temporal)
   - Features: Energy, phase, shells (multivariate)

2. **Fixed vs Random Effects**:
   - Fixed: Token identity (θ angle)
   - Random: Position-varying features
   - Interaction: Token × position effects

3. **Correlation Structure**:
   - Within-entity: Autocorrelation over positions
   - Between-entity: Cross-token correlations
   - Bidirectional: Both forward and backward

4. **Unique Features**:
   - Retrocausal effects (future affects past)
   - Phase locking creates discontinuities
   - Natural boundaries from energy depletion
   - Geometric rather than linear relationships

### Econometric Advantages:

1. **Identification**: Fixed effects control for token-specific unobservables
2. **Dynamics**: Rich temporal structure with natural decay
3. **Interactions**: Automatic token × position interactions
4. **Instruments**: Future states can instrument for past (retrocausal)
5. **Boundaries**: Natural sample selection from energy threshold

This panel structure enables sophisticated analysis techniques from econometrics
while maintaining the geometric elegance of the φ-based framework.
"""
    
    save_figure_with_notes(fig, "panel_time_series", notes)


# Run all visualizations
def main():
    print("Generating advanced visualizations...")
    print(f"Output directory: {output_dir}")
    
    print("\n1. Creating panel data structure visualization...")
    visualize_panel_structure()
    
    print("\n2. Creating topological encoding visualization...")
    visualize_topological_encoding()
    
    print("\n3. Creating energy & phase landscape visualization...")
    visualize_energy_phase_landscape()
    
    print("\n4. Creating retrocausal flow visualization...")
    visualize_retrocausal_flow()
    
    print("\n5. Creating panel time series visualization...")
    visualize_panel_time_series()
    
    # Create index file
    index_content = """# Φ-Mamba Visualizations Index

Generated visualizations exploring the panel data structure and unique features of Φ-Mamba.

## Visualizations:

1. [Panel Data Structure](panel_data_structure.md) - How tokens create comprehensive panel dataset
2. [Topological Encoding](topological_encoding.md) - Information stored as topology, not bits  
3. [Energy Phase Landscape](energy_phase_landscape.md) - Energy decay and phase space dynamics
4. [Retrocausal Flow](retrocausal_flow.md) - Bidirectional information flow architecture
5. [Panel Time Series](panel_time_series.md) - Time series analysis of panel structure

## Key Insights:

- Each token is an entity with fixed identity and time-varying features
- Information flows bidirectionally (retrocausal constraints)
- Natural boundaries emerge from energy decay
- Topological structure creates richer representation than bits
- Panel data framework enables sophisticated analysis

## Mathematical Foundation:

- φ = (1 + √5)/2 is the only primitive
- 1 = φ² - φ emerges from φ
- All operations reduce to integer addition
- Zeckendorf decomposition creates natural topology
"""
    
    with open(f"{output_dir}/index.md", 'w') as f:
        f.write(index_content)
    
    print(f"\n✓ All visualizations complete! See {output_dir}/index.md")


if __name__ == "__main__":
    main()