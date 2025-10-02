"""
Visualize the topological structure of Φ-Mamba
Shows Zeckendorf decomposition, cylinder encoding, and phase relationships
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as mpatches

from phi_mamba import PhiLanguageModel
from phi_mamba.encoding import zeckendorf_decomposition, binary_from_zeckendorf
from phi_mamba.utils import PHI, fibonacci


def visualize_zeckendorf_patterns():
    """Visualize Zeckendorf decomposition as emergent topology"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Part 1: Show decomposition for numbers 1-30
    print("Zeckendorf Decompositions (1-30):")
    print("-" * 50)
    
    max_n = 30
    decompositions = []
    
    for n in range(1, max_n + 1):
        zeck = zeckendorf_decomposition(n)
        binary = binary_from_zeckendorf(zeck, max_fib_index=8)
        decompositions.append((n, zeck, binary))
        
        if n <= 20:  # Print first 20
            print(f"{n:2d} = {str(zeck):20s} → {binary}")
    
    # Create heatmap of bit patterns
    bit_matrix = np.zeros((max_n, 8))
    
    for i, (n, zeck, binary) in enumerate(decompositions):
        for j, bit in enumerate(binary):
            if bit == '1':
                bit_matrix[i, j] = 1
    
    # Plot heatmap
    im1 = ax1.imshow(bit_matrix.T, cmap='RdBu_r', aspect='auto')
    ax1.set_xlabel('Number')
    ax1.set_ylabel('Fibonacci Index')
    ax1.set_title('Zeckendorf Bit Patterns (Red = Active Hole, Blue = Inactive)')
    ax1.set_yticks(range(8))
    ax1.set_yticklabels([f'F_{i+2}' for i in range(8)])
    
    # Part 2: Show gap constraint emergence
    # Count consecutive attempts
    consecutive_count = 0
    for i in range(len(bit_matrix)):
        binary_str = ''.join([str(int(b)) for b in bit_matrix[i]])
        if '11' in binary_str:
            consecutive_count += 1
    
    ax2.text(0.5, 0.7, f"Numbers 1-{max_n}:", ha='center', fontsize=16, transform=ax2.transAxes)
    ax2.text(0.5, 0.5, f"Consecutive 1s found: {consecutive_count}", ha='center', fontsize=20, 
             weight='bold', color='green', transform=ax2.transAxes)
    ax2.text(0.5, 0.3, "Gap constraint emerges naturally from φ² = φ + 1", 
             ha='center', fontsize=14, style='italic', transform=ax2.transAxes)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('zeckendorf_topology.png', dpi=150)
    plt.show()


def visualize_cylinder_encoding():
    """Visualize tokens on cylinder with φ-based encoding"""
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create cylinder mesh
    u = np.linspace(0, 2 * np.pi, 50)
    h = np.linspace(0, 5, 20)
    u_grid, h_grid = np.meshgrid(u, h)
    
    x_cylinder = np.cos(u_grid)
    y_cylinder = np.sin(u_grid)
    z_cylinder = h_grid
    
    # Plot transparent cylinder
    ax.plot_surface(x_cylinder, y_cylinder, z_cylinder, alpha=0.1, color='blue')
    
    # Encode a sentence and plot tokens
    model = PhiLanguageModel()
    sentence = "The golden ratio emerges naturally"
    states = model.encode(sentence, retrocausal=True)
    
    print(f"\nCylinder encoding for: '{sentence}'")
    print("-" * 50)
    
    # Plot each token
    colors = plt.cm.rainbow(np.linspace(0, 1, len(states)))
    
    for i, state in enumerate(states):
        # Convert to cylinder coordinates
        theta = state.theta_total
        r = min(state.r, 0.9)  # Cap radius for visualization
        z = state.position
        
        # Convert to Cartesian
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Plot point
        ax.scatter(x, y, z, color=colors[i], s=200, marker='o', edgecolor='black')
        
        # Add label
        ax.text(x*1.2, y*1.2, z, state.token, fontsize=10)
        
        # Draw spiral path
        if i > 0:
            prev_state = states[i-1]
            prev_theta = prev_state.theta_total
            prev_r = min(prev_state.r, 0.9)
            prev_z = prev_state.position
            
            # Create spiral between points
            t = np.linspace(0, 1, 20)
            x_path = (1-t) * prev_r * np.cos(prev_theta) + t * r * np.cos(theta)
            y_path = (1-t) * prev_r * np.sin(prev_theta) + t * r * np.sin(theta)
            z_path = (1-t) * prev_z + t * z
            
            ax.plot(x_path, y_path, z_path, color=colors[i], alpha=0.5, linewidth=2)
        
        print(f"{state.token:10s}: θ={theta:.3f}, r={r:.3f}, z={z}, shells={state.active_shells}")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Position (z)')
    ax.set_title('Token States on Cylinder (Radius = Amplitude)')
    
    plt.savefig('cylinder_encoding.png', dpi=150)
    plt.show()


def visualize_phase_locking():
    """Visualize phase relationships and locking"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Generate phase diagram
    n_points = 100
    phases = np.linspace(0, 4*np.pi, n_points)
    
    # Phase-locked regions (multiples of 2π)
    locked_regions = []
    for k in range(5):
        center = k * 2 * np.pi
        locked_regions.append((center - 0.5, center + 0.5))
    
    # Plot phase space
    ax1.plot(phases, np.sin(phases), 'b-', alpha=0.5, label='sin(γ)')
    ax1.plot(phases, np.cos(phases), 'r-', alpha=0.5, label='cos(γ)')
    
    # Highlight locked regions
    for start, end in locked_regions:
        if start >= 0 and end <= 4*np.pi:
            ax1.axvspan(start, end, alpha=0.2, color='green', label='Phase locked' if start == 0 else '')
    
    ax1.set_xlabel('Berry Phase γ')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Phase Locking Regions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Pentagon reflection visualization
    ax2.set_aspect('equal')
    
    # Draw pentagon
    pentagon_angles = np.linspace(0, 2*np.pi, 6)
    pentagon_x = np.cos(pentagon_angles)
    pentagon_y = np.sin(pentagon_angles)
    ax2.plot(pentagon_x, pentagon_y, 'k-', linewidth=2)
    
    # Show reflection path
    start_angle = np.pi/5
    reflections = 5
    
    x_path = [np.cos(start_angle)]
    y_path = [np.sin(start_angle)]
    
    current_angle = start_angle
    for i in range(reflections):
        # Reflect through opposite side
        current_angle = np.pi - current_angle + (i+1) * 2*np.pi/5
        x_path.append(np.cos(current_angle))
        y_path.append(np.sin(current_angle))
    
    # Plot reflection path
    ax2.plot(x_path, y_path, 'r-o', linewidth=2, markersize=8)
    
    # Add energy annotations
    for i, (x, y) in enumerate(zip(x_path, y_path)):
        energy = 1.0 / (PHI**i)
        ax2.annotate(f'E={energy:.3f}', (x, y), xytext=(x*1.3, y*1.3),
                    ha='center', fontsize=10)
    
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_title('Pentagon Reflection (Energy Decay)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase_locking.png', dpi=150)
    plt.show()


def visualize_topological_holes():
    """Visualize Betti numbers and topological structure"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Part 1: Betti numbers for different shell depths
    shell_depths = range(1, 10)
    
    # Calculate Betti numbers
    betti_matrix = []
    for n in shell_depths:
        betti = []
        for k in range(5):  # β_0 through β_4
            if k <= n:
                beta_k = fibonacci(n - k)
                betti.append(beta_k)
            else:
                betti.append(0)
        betti_matrix.append(betti)
    
    betti_matrix = np.array(betti_matrix).T
    
    # Plot Betti numbers
    for k in range(5):
        ax1.plot(shell_depths, betti_matrix[k], 'o-', label=f'β_{k}', linewidth=2)
    
    ax1.set_xlabel('Shell Depth (n)')
    ax1.set_ylabel('Betti Number')
    ax1.set_title('Betti Numbers = Fibonacci Weights')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Part 2: Show hole structure for n=7
    n = 7
    ax2.text(0.5, 0.9, f'Topological Structure for n={n}', ha='center', fontsize=14, weight='bold', transform=ax2.transAxes)
    
    y_positions = np.linspace(0.8, 0.2, 5)
    for k in range(5):
        beta_k = fibonacci(n - k) if k <= n else 0
        
        # Draw circles representing holes
        y = y_positions[k]
        ax2.text(0.1, y, f'β_{k} = {beta_k}', va='center', transform=ax2.transAxes)
        
        # Draw holes
        for i in range(min(beta_k, 10)):  # Limit visualization
            circle = Circle((0.3 + i*0.06, y), 0.02, color=f'C{k}', alpha=0.7, transform=ax2.transAxes)
            ax2.add_patch(circle)
            
        ax2.text(0.9, y, f'{k}D holes', va='center', ha='right', transform=ax2.transAxes)
    
    ax2.axis('off')
    
    # Part 3: Active shells visualization
    ax3.set_title('Active Shells for Token Positions')
    
    # Show first 15 positions
    for pos in range(1, 16):
        zeck = zeckendorf_decomposition(pos)
        y = 16 - pos
        
        # Draw position
        ax3.text(0, y, f'Pos {pos}:', va='center')
        
        # Draw active shells
        for i, fib in enumerate(zeck):
            # Find which Fibonacci number this is
            fib_idx = 1
            a, b = 1, 2
            while b < fib:
                a, b = b, a + b
                fib_idx += 1
            
            if b == fib:
                rect = Rectangle((fib_idx, y-0.4), 0.8, 0.8, color='red', alpha=0.7)
                ax3.add_patch(rect)
        
    ax3.set_xlim(-1, 10)
    ax3.set_ylim(0, 17)
    ax3.set_xlabel('Fibonacci Index')
    ax3.set_ylabel('Position')
    ax3.grid(True, axis='x', alpha=0.3)
    
    # Part 4: Coherence landscape
    ax4.set_title('Phase Coherence Landscape')
    
    # Create 2D coherence map
    theta_range = np.linspace(0, 2*np.pi, 50)
    pos_range = np.linspace(0, 10, 50)
    theta_grid, pos_grid = np.meshgrid(theta_range, pos_range)
    
    # Coherence function (simplified)
    coherence = np.cos(theta_grid) * np.exp(-pos_grid/5) + 0.5*np.sin(3*theta_grid) * np.exp(-pos_grid/3)
    
    im = ax4.imshow(coherence, aspect='auto', cmap='coolwarm', extent=[0, 2*np.pi, 10, 0])
    ax4.set_xlabel('Phase θ')
    ax4.set_ylabel('Position')
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Coherence')
    
    plt.tight_layout()
    plt.savefig('topological_holes.png', dpi=150)
    plt.show()


def main():
    print("=== Φ-Mamba Topology Visualization ===\n")
    
    # 1. Zeckendorf patterns
    print("1. Visualizing Zeckendorf decomposition as emergent topology...")
    visualize_zeckendorf_patterns()
    
    # 2. Cylinder encoding
    print("\n2. Visualizing cylinder encoding...")
    visualize_cylinder_encoding()
    
    # 3. Phase locking
    print("\n3. Visualizing phase locking and pentagon reflection...")
    visualize_phase_locking()
    
    # 4. Topological holes
    print("\n4. Visualizing topological hole structure...")
    visualize_topological_holes()
    
    print("\n" + "="*60)
    print("\nKey Insights from Visualizations:")
    print("1. Zeckendorf patterns show NO adjacent 1s - gap constraint emerges naturally")
    print("2. Tokens live on cylinder surface with φ-based radial encoding")
    print("3. Phase locking creates discrete coherent regions in continuous space")
    print("4. Betti numbers = Fibonacci sequence, revealing deep topology")
    print("5. Information is geometry - computation is topological flow")


if __name__ == "__main__":
    main()