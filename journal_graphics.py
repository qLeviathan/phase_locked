"""
Journal-Quality Graphics for Φ-Mamba arXiv Preprint
Generates publication-ready figures with proper formatting
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, Rectangle
from matplotlib.collections import PatchCollection
import seaborn as sns
from scipy import stats
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import os

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['text.usetex'] = False  # Set True if LaTeX is available
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Golden ratio constant
PHI = (1 + np.sqrt(5)) / 2

# Create output directory
output_dir = "journal_figures"
os.makedirs(output_dir, exist_ok=True)

def figure_1_theoretical_framework():
    """Figure 1: Theoretical Framework - Game Tree with φ-weights"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel: Extensive form game tree
    ax1.set_title("(a) Extensive Form Game with φ-Discounting", fontweight='bold')
    
    # Create game tree
    positions = {
        'root': (0.5, 0.9),
        'L1': (0.25, 0.65),
        'R1': (0.75, 0.65),
        'LL2': (0.1, 0.4),
        'LR2': (0.4, 0.4),
        'RL2': (0.6, 0.4),
        'RR2': (0.9, 0.4),
        'omega': (0.5, 0.15)
    }
    
    # Draw edges with φ weights
    edges = [
        ('root', 'L1', r'$\phi^{-1}$'),
        ('root', 'R1', r'$\phi^{-1}$'),
        ('L1', 'LL2', r'$\phi^{-2}$'),
        ('L1', 'LR2', r'$\phi^{-2}$'),
        ('R1', 'RL2', r'$\phi^{-2}$'),
        ('R1', 'RR2', r'$\phi^{-2}$'),
        ('LL2', 'omega', r'$\psi^{-1}$'),
        ('LR2', 'omega', r'$\psi^{-1}$'),
        ('RL2', 'omega', r'$\psi^{-1}$'),
        ('RR2', 'omega', r'$\psi^{-1}$')
    ]
    
    for start, end, label in edges:
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        
        # Draw edge
        ax1.plot([x1, x2], [y1, y2], 'k-', alpha=0.7, linewidth=1.5)
        
        # Add weight label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax1.text(mid_x, mid_y, label, fontsize=9, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Draw nodes
    for node, (x, y) in positions.items():
        if node == 'omega':
            # Terminal node
            ax1.add_patch(Circle((x, y), 0.05, facecolor='red', edgecolor='black', linewidth=2))
            ax1.text(x, y-0.08, r'$\Omega$', ha='center', fontsize=12, fontweight='bold')
        else:
            # Decision nodes
            ax1.add_patch(Circle((x, y), 0.03, facecolor='lightblue', edgecolor='black', linewidth=1.5))
    
    ax1.text(0.5, 0.95, "Token Selection", ha='center', fontsize=11)
    ax1.text(0.05, 0.05, "Backward Induction: " + r'$V_t = \max_a\{u_t(a) + \beta V_{t+1}\}$', 
             fontsize=10, style='italic')
    
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Right panel: DiD identification strategy
    ax2.set_title("(b) Difference-in-Differences with Fibonacci Treatment", fontweight='bold')
    
    # Create synthetic DiD data
    time_periods = np.arange(10)
    np.random.seed(42)
    
    # Control group (no F_5 in Zeckendorf)
    control_pre = 0.5 + 0.05 * time_periods[:5] + 0.02 * np.random.randn(5)
    control_post = 0.75 + 0.03 * (time_periods[5:] - 5) + 0.02 * np.random.randn(5)
    control = np.concatenate([control_pre, control_post])
    
    # Treatment group (has F_5 in Zeckendorf)
    treatment_effect = 0.2
    treated_pre = 0.48 + 0.05 * time_periods[:5] + 0.02 * np.random.randn(5)
    treated_post = 0.73 + treatment_effect + 0.03 * (time_periods[5:] - 5) + 0.02 * np.random.randn(5)
    treated = np.concatenate([treated_pre, treated_post])
    
    # Plot DiD
    ax2.plot(time_periods, control, 'o-', color='blue', label='Control (no $F_5$)', 
             markersize=6, linewidth=2)
    ax2.plot(time_periods, treated, 's-', color='red', label='Treatment (has $F_5$)', 
             markersize=6, linewidth=2)
    
    # Add vertical line for treatment time
    ax2.axvline(x=4.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.text(4.5, 0.4, 'Treatment\nTime', ha='center', fontsize=10, color='gray')
    
    # Add DiD annotation
    ax2.annotate('', xy=(7, control_post[2]), xytext=(7, treated_post[2]),
                arrowprops=dict(arrowstyle='<->', color='green', linewidth=2))
    ax2.text(7.2, (control_post[2] + treated_post[2])/2, 
             r'$\hat{\delta}_{DiD}$', fontsize=12, color='green', fontweight='bold')
    
    ax2.set_xlabel('Position $t$')
    ax2.set_ylabel('Outcome (Phase Coherence)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.3, 1.1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure_1_theoretical_framework.pdf", bbox_inches='tight')
    plt.savefig(f"{output_dir}/figure_1_theoretical_framework.png", bbox_inches='tight')
    plt.close()

def figure_2_equilibrium_dynamics():
    """Figure 2: Equilibrium Dynamics and Convergence"""
    fig = plt.figure(figsize=(12, 8))
    
    # Create 2x2 subplot
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # (a) Phase space trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("(a) Phase Space Trajectory", fontweight='bold')
    
    # Generate spiral trajectory
    t = np.linspace(0, 4*np.pi, 1000)
    r = np.exp(-t/(2*np.pi*PHI))
    x = r * np.cos(t)
    y = r * np.sin(t)
    
    # Color by time
    colors = plt.cm.viridis(t / max(t))
    for i in range(len(t)-1):
        ax1.plot(x[i:i+2], y[i:i+2], color=colors[i], linewidth=2)
    
    # Mark equilibrium points
    ax1.plot(0, 0, 'r*', markersize=15, label='Nash Equilibrium')
    ax1.plot(x[0], y[0], 'go', markersize=8, label='Initial State')
    
    ax1.set_xlabel(r'$\cos(\theta)$')
    ax1.set_ylabel(r'$\sin(\theta)$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # (b) Energy decay
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("(b) Energy Decay and Natural Termination", fontweight='bold')
    
    positions = np.arange(15)
    energy = PHI ** (-positions)
    termination_threshold = 0.01
    
    ax2.semilogy(positions, energy, 'b-', linewidth=2.5, label=r'$E(t) = \phi^{-t}$')
    ax2.axhline(y=termination_threshold, color='red', linestyle='--', 
                linewidth=2, label='Termination Threshold')
    ax2.fill_between(positions, 0, termination_threshold, alpha=0.2, color='red')
    
    # Mark termination point
    term_pos = np.where(energy < termination_threshold)[0][0]
    ax2.plot(term_pos, energy[term_pos], 'ro', markersize=10)
    ax2.annotate(f'Natural\nTermination\n(t={term_pos})', 
                xy=(term_pos, energy[term_pos]),
                xytext=(term_pos+2, energy[term_pos]*10),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    ax2.set_xlabel('Position $t$')
    ax2.set_ylabel('Energy $E(t)$')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_xlim(-0.5, 14.5)
    
    # (c) Mixed strategy equilibrium
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("(c) Mixed Strategy Nash Equilibrium", fontweight='bold')
    
    # Temperature parameter effect on strategy mixing
    temps = np.linspace(0.1, 2.0, 50)
    n_strategies = 5
    
    for i in range(n_strategies):
        # Utility differences
        u_i = np.sin(i * np.pi / 4) + 0.5
        
        # Quantal response probabilities
        probs = []
        for T in temps:
            exp_u = np.exp(u_i / T)
            total_exp = np.sum([np.exp((np.sin(j * np.pi / 4) + 0.5) / T) 
                               for j in range(n_strategies)])
            probs.append(exp_u / total_exp)
        
        ax3.plot(temps, probs, linewidth=2, label=f'Token {i+1}')
    
    ax3.set_xlabel('Temperature $T$')
    ax3.set_ylabel('Selection Probability')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.text(0.5, 0.9, r'$P_i = \frac{\exp(u_i/T)}{\sum_j \exp(u_j/T)}$', 
             transform=ax3.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    # (d) Bellman optimality
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    ax4.set_title("(d) Value Function Surface", fontweight='bold')
    
    # Create value function surface
    theta = np.linspace(0, 2*np.pi, 50)
    energy = np.linspace(0.01, 1, 50)
    THETA, ENERGY = np.meshgrid(theta, energy)
    
    # Value function V(θ, E) = E * cos(θ) + β * V_future
    VALUE = ENERGY * np.cos(THETA) * (1 + 1/PHI)
    
    # Plot surface
    surf = ax4.plot_surface(THETA, ENERGY, VALUE, cmap='viridis', 
                           alpha=0.8, edgecolor='none')
    
    # Add optimal path
    opt_theta = np.zeros(50)
    opt_energy = np.linspace(1, 0.01, 50)
    opt_value = opt_energy * (1 + 1/PHI)
    ax4.plot(opt_theta, opt_energy, opt_value, 'r-', linewidth=3, 
             label='Optimal Path')
    
    ax4.set_xlabel(r'Phase $\theta$')
    ax4.set_ylabel('Energy $E$')
    ax4.set_zlabel('Value $V(\theta, E)$')
    ax4.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure_2_equilibrium_dynamics.pdf", bbox_inches='tight')
    plt.savefig(f"{output_dir}/figure_2_equilibrium_dynamics.png", bbox_inches='tight')
    plt.close()

def figure_3_panel_data_structure():
    """Figure 3: Panel Data Structure and Econometric Properties"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) Panel data visualization
    ax1.set_title("(a) Token Panel Structure", fontweight='bold')
    
    # Create synthetic panel data
    n_entities = 8
    n_time = 10
    np.random.seed(42)
    
    # Entity fixed effects
    entity_effects = np.random.randn(n_entities) * 0.3
    
    # Time effects
    time_effects = np.linspace(0, 1, n_time)
    
    # Panel matrix
    panel_data = np.zeros((n_entities, n_time))
    for i in range(n_entities):
        for t in range(n_time):
            panel_data[i, t] = entity_effects[i] + time_effects[t] + 0.1 * np.random.randn()
            if t > 0:
                panel_data[i, t] += 0.5 * panel_data[i, t-1]  # AR(1) component
    
    # Heatmap
    im = ax1.imshow(panel_data, aspect='auto', cmap='RdBu_r')
    ax1.set_xlabel('Time (Position)')
    ax1.set_ylabel('Entity (Token ID)')
    ax1.set_xticks(range(n_time))
    ax1.set_yticks(range(n_entities))
    ax1.set_yticklabels([f'Token {i+1}' for i in range(n_entities)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Outcome', rotation=270, labelpad=15)
    
    # Add grid
    for i in range(n_entities + 1):
        ax1.axhline(i - 0.5, color='gray', linewidth=0.5)
    for t in range(n_time + 1):
        ax1.axvline(t - 0.5, color='gray', linewidth=0.5)
    
    # (b) Fixed vs Random Effects
    ax2.set_title("(b) Fixed vs Random Effects Decomposition", fontweight='bold')
    
    # Generate example data
    tokens = ['the', 'cat', 'sat', 'on', 'mat']
    fixed_effects = np.array([0.2, -0.1, 0.3, 0.0, 0.15])
    
    x = np.arange(len(tokens))
    width = 0.35
    
    # Fixed effects
    rects1 = ax2.bar(x - width/2, fixed_effects, width, label='Fixed Effects (θ)',
                     color='steelblue', alpha=0.8)
    
    # Random effects (time-varying)
    random_effects = np.random.randn(len(tokens)) * 0.15
    rects2 = ax2.bar(x + width/2, random_effects, width, label='Random Effects',
                     color='coral', alpha=0.8)
    
    ax2.set_xlabel('Token')
    ax2.set_ylabel('Effect Size')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tokens)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    
    # (c) Zeckendorf as natural experiment
    ax3.set_title("(c) Zeckendorf Decomposition as Natural Experiment", fontweight='bold')
    
    positions = range(1, 50)
    has_F5 = []
    has_F7 = []
    
    for pos in positions:
        # Compute Zeckendorf
        n = pos
        fibs = [1, 1]
        while fibs[-1] < n:
            fibs.append(fibs[-1] + fibs[-2])
        
        decomp = []
        for f in reversed(fibs[2:]):  # Skip first two 1s
            if f <= n:
                decomp.append(f)
                n -= f
        
        has_F5.append(1 if 5 in decomp else 0)
        has_F7.append(1 if 13 in decomp else 0)
    
    # Plot treatment assignment
    ax3.scatter(positions, has_F5, c='blue', marker='s', s=30, alpha=0.7, label='Has $F_5$ (5)')
    ax3.scatter(positions, np.array(has_F7) + 0.05, c='red', marker='^', s=30, alpha=0.7, label='Has $F_7$ (13)')
    
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Treatment Status')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Control', 'Treated'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.text(25, 0.5, 'Exogenous\nVariation', ha='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    # (d) Regression coefficients
    ax4.set_title("(d) Econometric Model Coefficients", fontweight='bold')
    
    # Simulated regression results
    coef_names = ['Intercept', 'Treatment', 'Time', 'Treatment×Time', 'Energy', 'Phase']
    coef_values = np.array([0.5, 0.25, 0.1, 0.15, -0.3, 0.2])
    std_errors = np.array([0.05, 0.08, 0.03, 0.06, 0.1, 0.07])
    
    # Forest plot
    y_pos = np.arange(len(coef_names))
    ax4.errorbar(coef_values, y_pos, xerr=1.96*std_errors, fmt='o', 
                color='darkgreen', markersize=8, capsize=5, linewidth=2)
    
    # Add vertical line at zero
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    
    # Significant coefficients
    for i, (coef, se) in enumerate(zip(coef_values, std_errors)):
        if abs(coef) > 1.96 * se:
            ax4.plot(coef, i, 'o', color='darkgreen', markersize=10)
        else:
            ax4.plot(coef, i, 'o', color='gray', markersize=8)
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(coef_names)
    ax4.set_xlabel('Coefficient Estimate')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.text(0.02, 5.5, '95% CI', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure_3_panel_data_structure.pdf", bbox_inches='tight')
    plt.savefig(f"{output_dir}/figure_3_panel_data_structure.png", bbox_inches='tight')
    plt.close()

def figure_4_mechanism_design():
    """Figure 4: Mechanism Design and Implementation"""
    fig = plt.figure(figsize=(14, 6))
    
    # Create custom layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3, height_ratios=[1.5, 1])
    
    # Main diagram: Full mechanism
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_title("Φ-Mamba as Dynamic Mechanism Design", fontweight='bold', fontsize=16)
    
    # Components of the mechanism
    components = {
        'Input': (0.1, 0.5),
        'Encoder': (0.25, 0.5),
        'Game': (0.5, 0.5),
        'Decoder': (0.75, 0.5),
        'Output': (0.9, 0.5)
    }
    
    # Draw components
    for comp, (x, y) in components.items():
        if comp in ['Input', 'Output']:
            # Circular nodes
            circle = Circle((x, y), 0.06, facecolor='lightcoral', edgecolor='black', linewidth=2)
            ax_main.add_patch(circle)
        else:
            # Rectangular boxes
            width, height = 0.12, 0.2
            rect = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                 boxstyle="round,pad=0.02", 
                                 facecolor='lightblue', edgecolor='black', linewidth=2)
            ax_main.add_patch(rect)
        
        ax_main.text(x, y, comp, ha='center', va='center', fontweight='bold', fontsize=11)
    
    # Draw connections with labels
    connections = [
        ('Input', 'Encoder', r'$\theta = h(\text{token})$'),
        ('Encoder', 'Game', r'$s_t = (\theta, E_t, \phi_t)$'),
        ('Game', 'Decoder', r'$a^* = \arg\max V(s)$'),
        ('Decoder', 'Output', r'$\text{token} = h^{-1}(\theta^*)$')
    ]
    
    for start, end, label in connections:
        x1, y1 = components[start]
        x2, y2 = components[end]
        
        # Arrow
        ax_main.annotate('', xy=(x2-0.08, y2), xytext=(x1+0.08, y1),
                        arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax_main.text(mid_x, mid_y + 0.1, label, ha='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add backward flow
    ax_main.annotate('', xy=(0.25, 0.3), xytext=(0.75, 0.3),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='dashed'))
    ax_main.text(0.5, 0.25, 'Retrocausal Constraint', ha='center', color='red', fontsize=10)
    
    # Add formulas
    ax_main.text(0.5, 0.85, r'Mechanism: $\mathcal{M} = \langle S, A, g, h \rangle$', 
                ha='center', fontsize=12, style='italic')
    ax_main.text(0.5, 0.05, r'Implementation: $a^*(s) \in \arg\max_{a \in A} u(a, s) + \beta \cdot V(g(s, a))$',
                ha='center', fontsize=11)
    
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.axis('off')
    
    # Bottom panels
    # (a) Incentive Compatibility
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_title("(a) Incentive Compatibility", fontsize=12)
    
    # Truth-telling equilibrium
    true_values = np.linspace(0, 1, 100)
    reported_values_IC = true_values  # Truth-telling
    reported_values_not_IC = 0.5 * true_values + 0.25  # Distorted
    
    ax1.plot(true_values, reported_values_IC, 'g-', linewidth=2, label='IC Mechanism')
    ax1.plot(true_values, reported_values_not_IC, 'r--', linewidth=2, label='Non-IC')
    ax1.plot(true_values, true_values, 'k:', alpha=0.5, label='Truth-telling')
    
    ax1.set_xlabel('True Type θ')
    ax1.set_ylabel('Reported Type')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # (b) Efficiency
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_title("(b) Allocative Efficiency", fontsize=12)
    
    # Pareto frontier
    theta = np.linspace(0, np.pi/2, 100)
    x_pareto = np.cos(theta)
    y_pareto = np.sin(theta)
    
    ax2.fill_between(x_pareto, 0, y_pareto, alpha=0.2, color='lightblue', label='Feasible Set')
    ax2.plot(x_pareto, y_pareto, 'b-', linewidth=2, label='Pareto Frontier')
    
    # Mark φ-optimal point
    phi_angle = np.arctan(1/PHI)
    x_phi = np.cos(phi_angle)
    y_phi = np.sin(phi_angle)
    ax2.plot(x_phi, y_phi, 'ro', markersize=10, label='φ-Optimal')
    
    ax2.set_xlabel('Player 1 Utility')
    ax2.set_ylabel('Player 2 Utility')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # (c) Revenue Equivalence
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_title("(c) Revenue Properties", fontsize=12)
    
    # Different auction formats
    reserve_prices = np.linspace(0, 1, 50)
    revenue_first_price = 0.5 * (1 - reserve_prices**2)
    revenue_second_price = 0.5 * (1 - reserve_prices**2)
    revenue_phi = (1/PHI) * (1 - reserve_prices**PHI)
    
    ax3.plot(reserve_prices, revenue_first_price, 'b-', linewidth=2, label='First-Price')
    ax3.plot(reserve_prices, revenue_second_price, 'g--', linewidth=2, label='Second-Price')
    ax3.plot(reserve_prices, revenue_phi, 'r-', linewidth=2.5, label='φ-Mechanism')
    
    ax3.set_xlabel('Reserve Price')
    ax3.set_ylabel('Expected Revenue')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure_4_mechanism_design.pdf", bbox_inches='tight')
    plt.savefig(f"{output_dir}/figure_4_mechanism_design.png", bbox_inches='tight')
    plt.close()

def create_all_figures():
    """Generate all journal-quality figures"""
    print("Generating journal-quality figures...")
    
    print("Creating Figure 1: Theoretical Framework...")
    figure_1_theoretical_framework()
    
    print("Creating Figure 2: Equilibrium Dynamics...")
    figure_2_equilibrium_dynamics()
    
    print("Creating Figure 3: Panel Data Structure...")
    figure_3_panel_data_structure()
    
    print("Creating Figure 4: Mechanism Design...")
    figure_4_mechanism_design()
    
    print(f"\nAll figures saved to {output_dir}/")
    print("Formats: PDF (for LaTeX) and PNG (for preview)")

if __name__ == "__main__":
    create_all_figures()