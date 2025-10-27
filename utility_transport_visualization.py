"""
High-quality visualization of utility transport system with Laplacian dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2
PSI = -1/PHI

def fibonacci(n):
    """Generate Fibonacci number"""
    if n <= 1:
        return n
    return int((PHI**n - PSI**n) / np.sqrt(5))

def zeckendorf_decomposition(n):
    """Decompose n into non-adjacent Fibonacci numbers"""
    if n == 0:
        return []
    fibs = [1, 2]
    while fibs[-1] < n:
        fibs.append(fibs[-1] + fibs[-2])
    
    result = []
    i = len(fibs) - 1
    while i >= 0 and n > 0:
        if fibs[i] <= n:
            result.append(fibs[i])
            n -= fibs[i]
            i -= 2  # Skip adjacent
        else:
            i -= 1
    return result

def create_utility_surface_plot():
    """Visualize the utility surface evolution"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid
    t = np.linspace(0, 10, 100)
    positions = np.linspace(0, 20, 100)
    T, P = np.meshgrid(t, positions)
    
    # Utility function with decay
    Terminal_time = 10
    U = PHI**(Terminal_time - T) * np.exp(-P/10) * (1 + 0.5*np.sin(P))
    
    # Create surface
    surf = ax.plot_surface(T, P, U, cmap='coolwarm', alpha=0.8,
                          linewidth=0, antialiased=True)
    
    # Add contours
    contours = ax.contour(T, P, U, zdir='z', offset=0, cmap='coolwarm', alpha=0.5)
    
    # Labels
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Position', fontsize=12)
    ax.set_zlabel('Utility', fontsize=12)
    ax.set_title('Utility Surface: High → Unity', fontsize=16, pad=20)
    
    # Add annotations
    ax.text(0, 10, PHI**10, 'Initial\nHigh Utility', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    ax.text(10, 10, 1, 'Terminal Ω\nUnity', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7))
    
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()
    return fig

def create_laplacian_transport_network():
    """Visualize token graph with Laplacian transport"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Create token network
    G = nx.Graph()
    n_tokens = 8
    
    # Add nodes with positions
    pos = {}
    for i in range(n_tokens):
        angle = 2 * np.pi * i / n_tokens
        pos[i] = (np.cos(angle), np.sin(angle))
        G.add_node(i, utility=PHI**(n_tokens-i))
    
    # Add edges with φ-weights
    for i in range(n_tokens):
        for j in range(i+1, n_tokens):
            weight = PHI**(-abs(i-j))
            if weight > 0.1:  # Only strong connections
                G.add_edge(i, j, weight=weight)
    
    # Plot 1: Network structure
    ax1.set_title('Token Network with φ-Coupling', fontsize=14)
    
    # Draw network
    utilities = [G.nodes[i]['utility'] for i in G.nodes()]
    node_colors = plt.cm.Reds(np.array(utilities) / max(utilities))
    
    nx.draw(G, pos, ax=ax1, node_color=node_colors, node_size=1000,
            with_labels=True, font_size=12, font_weight='bold')
    
    # Draw edge weights
    edge_weights = nx.get_edge_attributes(G, 'weight')
    for (i, j), w in edge_weights.items():
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        ax1.text((x1+x2)/2, (y1+y2)/2, f'{w:.2f}', 
                fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Plot 2: Utility flow over time
    ax2.set_title('Utility Transport Dynamics', fontsize=14)
    
    time_steps = 50
    dt = 0.1
    utilities_over_time = np.zeros((n_tokens, time_steps))
    
    # Initialize
    U = np.array([G.nodes[i]['utility'] for i in range(n_tokens)])
    utilities_over_time[:, 0] = U
    
    # Compute Laplacian
    L = nx.laplacian_matrix(G, weight='weight').toarray()
    
    # Simulate transport
    for t in range(1, time_steps):
        # Laplacian transport with source and sink
        dU_dt = -L @ U + 0.1 * np.random.randn(n_tokens)  # Small noise as coherence
        dU_dt -= U / PHI  # Energy sink
        
        U = U + dt * dU_dt
        U = np.maximum(U, 0)  # Keep positive
        utilities_over_time[:, t] = U
    
    # Plot evolution
    for i in range(n_tokens):
        ax2.plot(utilities_over_time[i, :], label=f'Token {i}', linewidth=2)
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Utility', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_did_gating_visualization():
    """Visualize the DiD gating mechanism"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gating pattern visualization
    positions = list(range(1, 25))
    n_positions = len(positions)
    
    # Create gate patterns for different Fibonacci scales
    scales = [2, 3, 5, 8]
    colors = ['red', 'blue', 'green', 'purple']
    
    ax1.set_title('Hierarchical DiD Gating System', fontsize=16)
    ax1.set_xlim(0, n_positions + 1)
    ax1.set_ylim(-0.5, len(scales) + 0.5)
    
    for i, (fib, color) in enumerate(zip(scales, colors)):
        y_pos = len(scales) - i - 1
        ax1.text(-0.5, y_pos, f'F_{{{fib}}}', fontsize=12, ha='right', va='center')
        
        for pos in positions:
            zeck = zeckendorf_decomposition(pos)
            if fibonacci(fib) in zeck:
                # Open gate
                rect = patches.Rectangle((pos-0.4, y_pos-0.3), 0.8, 0.6, 
                                       linewidth=2, edgecolor=color, 
                                       facecolor=color, alpha=0.3)
                ax1.add_patch(rect)
                ax1.text(pos, y_pos, 'O', ha='center', va='center', 
                        fontsize=10, weight='bold')
            else:
                # Closed gate
                ax1.text(pos, y_pos, 'X', ha='center', va='center', 
                        fontsize=10, color='gray')
    
    ax1.set_xlabel('Position', fontsize=12)
    ax1.set_ylabel('Fibonacci Scale', fontsize=12)
    ax1.set_xticks(positions[::2])
    ax1.set_yticks([])
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Utility flow with gating
    ax2.set_title('Gated Utility Transport', fontsize=16)
    
    # Simulate utility with gates
    utility = PHI**np.arange(n_positions, 0, -1)
    gated_utility = utility.copy()
    
    # Apply gates (example with F_3)
    f3_positions = [i for i, pos in enumerate(positions) if fibonacci(3) in zeckendorf_decomposition(pos)]
    
    x = np.array(positions)
    ax2.fill_between(x, 0, utility, alpha=0.3, color='gray', label='Potential Utility')
    
    # Show gated regions
    for i in range(len(positions)):
        if i in f3_positions:
            ax2.bar(positions[i], gated_utility[i], width=0.8, 
                   color='green', alpha=0.7, edgecolor='darkgreen', linewidth=2)
        else:
            ax2.bar(positions[i], gated_utility[i] * 0.3, width=0.8, 
                   color='red', alpha=0.7, edgecolor='darkred', linewidth=2)
    
    # Add flow arrows
    for i in range(len(positions) - 1):
        if i in f3_positions and i+1 in f3_positions:
            ax2.annotate('', xy=(positions[i+1], gated_utility[i+1]), 
                        xytext=(positions[i], gated_utility[i]),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    ax2.set_xlabel('Position', fontsize=12)
    ax2.set_ylabel('Utility', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_complete_system_diagram():
    """Create a complete system flow diagram"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Φ-Mamba Utility Transport System', 
            fontsize=20, ha='center', weight='bold')
    
    # Components
    components = [
        {'pos': (5, 8), 'text': 'Initial State\nU = φ^T', 'color': 'red'},
        {'pos': (5, 6.5), 'text': 'Laplacian Transport\n∂U/∂t = -Δ_φ U', 'color': 'blue'},
        {'pos': (3, 5), 'text': 'DiD Gates\nTreatment', 'color': 'green'},
        {'pos': (7, 5), 'text': 'Berry Phase\nCoherence', 'color': 'purple'},
        {'pos': (5, 3.5), 'text': 'Energy Sink\nU → U/φ', 'color': 'orange'},
        {'pos': (5, 2), 'text': 'Terminal State\nU = 1 (Unity)', 'color': 'darkgreen'},
    ]
    
    # Draw components
    for comp in components:
        x, y = comp['pos']
        rect = patches.FancyBboxPatch((x-1.2, y-0.4), 2.4, 0.8,
                                     boxstyle="round,pad=0.1",
                                     facecolor=comp['color'], alpha=0.3,
                                     edgecolor=comp['color'], linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, comp['text'], ha='center', va='center', 
               fontsize=12, weight='bold')
    
    # Draw arrows
    arrows = [
        ((5, 7.6), (5, 6.9)),  # Initial → Laplacian
        ((5, 6.1), (5, 3.9)),  # Laplacian → Sink
        ((3, 4.6), (4.5, 3.7)),  # Gates → Sink
        ((7, 4.6), (5.5, 3.7)),  # Coherence → Sink
        ((5, 3.1), (5, 2.4)),  # Sink → Terminal
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add mathematical annotations
    ax.text(1, 7, 'Zeckendorf\nDecomposition', fontsize=10, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    ax.text(9, 7, 'φ-Geometry\nCoupling', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="cyan", alpha=0.5))
    
    # Add flow equation
    ax.text(5, 0.5, 'dU/dt = -Δ_φU × Gates + Coherence - U/φ', 
           fontsize=14, ha='center', style='italic',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    return fig

# Generate all visualizations
if __name__ == "__main__":
    # Create output directory
    import os
    output_dir = "utility_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save all figures
    print("Generating utility surface plot...")
    fig1 = create_utility_surface_plot()
    fig1.savefig(f"{output_dir}/utility_surface.png", dpi=300, bbox_inches='tight')
    
    print("Generating Laplacian transport network...")
    fig2 = create_laplacian_transport_network()
    fig2.savefig(f"{output_dir}/laplacian_transport.png", dpi=300, bbox_inches='tight')
    
    print("Generating DiD gating visualization...")
    fig3 = create_did_gating_visualization()
    fig3.savefig(f"{output_dir}/did_gating.png", dpi=300, bbox_inches='tight')
    
    print("Generating complete system diagram...")
    fig4 = create_complete_system_diagram()
    fig4.savefig(f"{output_dir}/complete_system.png", dpi=300, bbox_inches='tight')
    
    print(f"All visualizations saved to {output_dir}/")
    plt.show()