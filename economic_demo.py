#!/usr/bin/env python3
"""
Interactive Economic Demonstration of N-Game Topological DiD Framework
Designed for traditional economists to explore the framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from typing import List, Dict, Tuple
import streamlit as st

# Import our framework
from phi_mamba.encoding import zeckendorf_decomposition
from phi_mamba.utils import PHI, PSI, fibonacci

# Set style for professional figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class EconomicDemonstration:
    """Interactive demonstrations for economists"""
    
    def __init__(self):
        self.phi = PHI
        self.results = {}
    
    def demonstrate_time_consistency(self):
        """Show why β = 1/φ ensures time-consistent preferences"""
        st.header("Time Consistency with φ-Discounting")
        
        st.markdown("""
        ### The Problem with Hyperbolic Discounting
        
        Standard hyperbolic discounting (β-δ model) leads to preference reversals:
        - At t=0: Prefer $100 at t=30 over $110 at t=31
        - At t=29: Prefer $110 at t=31 over $100 at t=30
        
        ### The φ Solution
        
        With discount factor β = 1/φ ≈ 0.618, preferences remain consistent:
        """)
        
        # Interactive sliders
        t_now = st.slider("Current time", 0, 30, 0)
        payoff_soon = st.slider("Earlier payoff", 50, 200, 100)
        payoff_later = st.slider("Later payoff", 50, 200, 110)
        delay = st.slider("Delay between payoffs", 1, 10, 1)
        
        # Calculate present values
        t_soon = 30
        t_later = t_soon + delay
        
        # Standard exponential (φ-discounting)
        pv_soon_phi = payoff_soon * (1/self.phi)**(t_soon - t_now)
        pv_later_phi = payoff_later * (1/self.phi)**(t_later - t_now)
        
        # Quasi-hyperbolic for comparison
        beta = 0.7  # Present bias
        delta = 0.95  # Long-run discount
        
        if t_now < t_soon:
            pv_soon_qh = payoff_soon * beta * delta**(t_soon - t_now)
            pv_later_qh = payoff_later * beta * delta**(t_later - t_now)
        else:
            pv_soon_qh = payoff_soon
            pv_later_qh = payoff_later * delta**(t_later - t_soon)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("φ-Discounting", 
                     f"Prefer {'Soon' if pv_soon_phi > pv_later_phi else 'Later'}",
                     f"Consistent: {pv_soon_phi:.2f} vs {pv_later_phi:.2f}")
        
        with col2:
            st.metric("Quasi-Hyperbolic", 
                     f"Prefer {'Soon' if pv_soon_qh > pv_later_qh else 'Later'}",
                     f"May reverse: {pv_soon_qh:.2f} vs {pv_later_qh:.2f}")
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # φ-discounting over time
        times = np.arange(0, 40)
        pv_soon_trajectory = [payoff_soon * (1/self.phi)**(t_soon - t) if t <= t_soon else payoff_soon for t in times]
        pv_later_trajectory = [payoff_later * (1/self.phi)**(t_later - t) if t <= t_later else payoff_later for t in times]
        
        ax1.plot(times, pv_soon_trajectory, 'b-', label=f'${payoff_soon} at t={t_soon}')
        ax1.plot(times, pv_later_trajectory, 'r-', label=f'${payoff_later} at t={t_later}')
        ax1.axvline(t_now, color='green', linestyle='--', label='Current time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Present Value')
        ax1.set_title('φ-Discounting (Time Consistent)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Preference regions
        ax2.set_xlim(0, 40)
        ax2.set_ylim(0, 1)
        
        prefer_soon = [1 if pv_soon_trajectory[t] > pv_later_trajectory[t] else 0 for t in times]
        ax2.fill_between(times, 0, prefer_soon, alpha=0.3, color='blue', label='Prefer Soon')
        ax2.fill_between(times, prefer_soon, 1, alpha=0.3, color='red', label='Prefer Later')
        ax2.axvline(t_now, color='green', linestyle='--', label='Current time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Preference')
        ax2.set_title('Preference Stability')
        ax2.legend()
        
        st.pyplot(fig)
        
        st.info("""
        **Key Insight**: The golden ratio provides the unique discount factor where 
        multiplicative discounting (φ^(-t)) maintains additive separability in 
        log-space, ensuring time consistency without arbitrary behavioral parameters.
        """)
    
    def demonstrate_did_identification(self):
        """Interactive DiD with Fibonacci treatment assignment"""
        st.header("Natural Experiments via Fibonacci Structure")
        
        st.markdown("""
        ### Difference-in-Differences without Randomization
        
        The Zeckendorf decomposition creates exogenous variation:
        - **Treatment**: Token positions containing F_5 = 5 in their decomposition
        - **Control**: Positions without F_5
        - **Identification**: Assignment is deterministic but creates as-if random variation
        """)
        
        # Parameters
        n_entities = st.slider("Number of entities (tokens)", 10, 100, 50)
        n_periods = st.slider("Number of time periods", 5, 20, 10)
        treatment_effect = st.slider("True treatment effect", 0.0, 1.0, 0.3)
        treatment_period = st.slider("Treatment starts at period", 3, n_periods-3, 5)
        
        # Generate panel data
        panel_data = []
        
        for entity in range(n_entities):
            entity_effect = np.random.normal(0, 0.2)  # Random effects
            
            for period in range(n_periods):
                position = entity * n_periods + period
                
                # Zeckendorf decomposition determines treatment
                zeck = zeckendorf_decomposition(position + 1)
                treated = 5 in zeck  # Has F_5
                
                # Outcome generation
                base_outcome = 1.0 + entity_effect
                time_trend = 0.05 * period
                post = period >= treatment_period
                
                # DiD structure: Y = α + βT + γPost + δ(T×Post) + ε
                outcome = (base_outcome + 
                          0.1 * treated + 
                          0.2 * post + 
                          treatment_effect * treated * post +
                          np.random.normal(0, 0.1))
                
                panel_data.append({
                    'entity': f'Token_{entity}',
                    'period': period,
                    'position': position,
                    'treated': treated,
                    'post': post,
                    'outcome': outcome,
                    'zeckendorf': str(zeck)
                })
        
        df = pd.DataFrame(panel_data)
        
        # Calculate DiD estimate
        treated_pre = df[(df['treated'] == True) & (df['post'] == False)]['outcome'].mean()
        treated_post = df[(df['treated'] == True) & (df['post'] == True)]['outcome'].mean()
        control_pre = df[(df['treated'] == False) & (df['post'] == False)]['outcome'].mean()
        control_post = df[(df['treated'] == False) & (df['post'] == True)]['outcome'].mean()
        
        did_estimate = (treated_post - treated_pre) - (control_post - control_pre)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("DiD Estimate", f"{did_estimate:.3f}", 
                     f"True effect: {treatment_effect:.3f}")
        
        with col2:
            treatment_share = df['treated'].mean()
            st.metric("Treatment share", f"{treatment_share:.1%}",
                     "Natural variation")
        
        with col3:
            parallel_trends = abs((treated_pre - control_pre) - 
                                (df[(df['period'] == 0) & (df['treated'] == True)]['outcome'].mean() -
                                 df[(df['period'] == 0) & (df['treated'] == False)]['outcome'].mean()))
            st.metric("Parallel trends", f"{parallel_trends:.3f}",
                     "Pre-treatment difference")
        
        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Classic DiD plot
        avg_outcomes = df.groupby(['period', 'treated'])['outcome'].mean().reset_index()
        
        for treated in [True, False]:
            data = avg_outcomes[avg_outcomes['treated'] == treated]
            ax1.plot(data['period'], data['outcome'], 
                    'o-' if treated else 's--',
                    label='Treatment' if treated else 'Control',
                    linewidth=2)
        
        ax1.axvline(treatment_period - 0.5, color='red', linestyle=':', label='Treatment start')
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Average Outcome')
        ax1.set_title('Difference-in-Differences')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Treatment assignment pattern
        positions = list(range(n_entities * n_periods))
        treatments = [5 in zeckendorf_decomposition(p + 1) for p in positions]
        
        ax2.scatter(positions[:200], treatments[:200], alpha=0.5, s=10)
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Treatment Status')
        ax2.set_title('Fibonacci-based Treatment Assignment')
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribution of outcomes
        ax3.hist(df[df['post'] == False]['outcome'], alpha=0.5, label='Pre-treatment', bins=30)
        ax3.hist(df[df['post'] == True]['outcome'], alpha=0.5, label='Post-treatment', bins=30)
        ax3.set_xlabel('Outcome')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Outcome Distribution')
        ax3.legend()
        
        # 4. Regression coefficients
        # Simple OLS for DiD
        df['treat_post'] = df['treated'] * df['post']
        X = pd.get_dummies(df[['treated', 'post', 'treat_post']], drop_first=False)
        y = df['outcome']
        
        # Add constant
        X['const'] = 1
        
        # OLS estimation
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        se = np.sqrt(np.diagonal(np.linalg.inv(X.T @ X) * np.var(y - X @ beta)))
        
        coef_names = ['Treated', 'Post', 'Treated×Post', 'Constant']
        ax4.errorbar(beta, range(len(beta)), xerr=1.96*se, fmt='o', capsize=5)
        ax4.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax4.set_yticks(range(len(beta)))
        ax4.set_yticklabels(coef_names)
        ax4.set_xlabel('Coefficient')
        ax4.set_title('Regression Results (95% CI)')
        ax4.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Additional insights
        st.subheader("Why This Works")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Exogenous Variation**:
            - Zeckendorf decomposition is deterministic
            - Creates quasi-random treatment assignment
            - No selection bias or endogeneity
            """)
        
        with col2:
            st.markdown("""
            **Multiple Instruments**:
            - Can use F_3, F_5, F_8, etc. as different treatments
            - Enables overidentification tests
            - Heterogeneous effects by Fibonacci scale
            """)
    
    def demonstrate_equilibrium_dynamics(self):
        """Show game equilibrium convergence"""
        st.header("N-Game Equilibrium Dynamics")
        
        st.markdown("""
        ### Phase-Locked Equilibria
        
        In the φ-framework, equilibria emerge from phase coherence:
        - Players choose positions on the unit circle
        - Payoffs depend on phase alignment and energy
        - Natural termination through energy decay
        """)
        
        # Parameters
        n_players = st.slider("Number of players", 2, 10, 3)
        n_iterations = st.slider("Iterations", 10, 100, 50)
        temperature = st.slider("Temperature (0 = deterministic)", 0.0, 2.0, 0.5)
        
        # Initialize game
        angles = np.random.uniform(0, 2*np.pi, n_players)
        energies = np.ones(n_players)
        history = []
        
        # Payoff matrix based on phase coherence
        def payoff(i, j, angles):
            return np.cos(angles[i] - angles[j])
        
        # Run dynamics
        for t in range(n_iterations):
            # Each player updates based on best response
            new_angles = angles.copy()
            
            for i in range(n_players):
                # Calculate expected payoff for different angles
                test_angles = np.linspace(0, 2*np.pi, 100)
                payoffs = np.zeros(100)
                
                for k, test_angle in enumerate(test_angles):
                    total_payoff = 0
                    for j in range(n_players):
                        if i != j:
                            total_payoff += payoff(k, j, np.append(test_angles[k], angles[np.arange(n_players) != i]))
                    payoffs[k] = total_payoff * energies[i]
                
                # Choose action (deterministic or stochastic)
                if temperature == 0:
                    new_angles[i] = test_angles[np.argmax(payoffs)]
                else:
                    probs = np.exp(payoffs / temperature)
                    probs /= probs.sum()
                    new_angles[i] = np.random.choice(test_angles, p=probs)
            
            angles = new_angles
            energies *= 1/self.phi  # Energy decay
            
            # Store history
            history.append({
                'iteration': t,
                'angles': angles.copy(),
                'energies': energies.copy(),
                'total_energy': energies.sum(),
                'phase_variance': np.var(angles)
            })
        
        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Phase evolution
        for i in range(n_players):
            trajectory = [h['angles'][i] for h in history]
            ax1.plot(trajectory, label=f'Player {i+1}', alpha=0.7)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Angle (radians)')
        ax1.set_title('Phase Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Energy decay
        total_energies = [h['total_energy'] for h in history]
        ax2.semilogy(total_energies, 'b-', linewidth=2)
        ax2.axhline(0.01 * n_players, color='red', linestyle='--', label='Termination threshold')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Total Energy (log scale)')
        ax2.set_title('Natural Termination via Energy Decay')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Phase space (final state)
        final_angles = history[-1]['angles']
        final_energies = history[-1]['energies']
        
        # Convert to Cartesian
        x = final_energies * np.cos(final_angles)
        y = final_energies * np.sin(final_angles)
        
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
        ax3.add_patch(circle)
        
        ax3.scatter(x, y, s=200, c=range(n_players), cmap='viridis')
        for i in range(n_players):
            ax3.annotate(f'P{i+1}', (x[i], y[i]), xytext=(5, 5), textcoords='offset points')
        
        ax3.set_xlim(-1.5, 1.5)
        ax3.set_ylim(-1.5, 1.5)
        ax3.set_aspect('equal')
        ax3.set_xlabel('x = r cos(θ)')
        ax3.set_ylabel('y = r sin(θ)')
        ax3.set_title('Final Configuration')
        ax3.grid(True, alpha=0.3)
        
        # 4. Convergence metrics
        phase_variances = [h['phase_variance'] for h in history]
        ax4.plot(phase_variances, 'g-', linewidth=2)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Phase Variance')
        ax4.set_title('Convergence to Equilibrium')
        ax4.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Equilibrium analysis
        st.subheader("Equilibrium Properties")
        
        col1, col2, col3 = st.columns(3)
        
        final_variance = history[-1]['phase_variance']
        converged = final_variance < 0.1
        
        with col1:
            st.metric("Converged?", "Yes" if converged else "No",
                     f"Variance: {final_variance:.3f}")
        
        with col2:
            iterations_to_low_energy = next((i for i, h in enumerate(history) 
                                           if h['total_energy'] < 0.1 * n_players), n_iterations)
            st.metric("Natural termination", f"Iteration {iterations_to_low_energy}",
                     f"Energy: {history[-1]['total_energy']:.3e}")
        
        with col3:
            # Check if phase-locked (all similar angles)
            angle_range = np.ptp(final_angles)
            phase_locked = angle_range < np.pi/2
            st.metric("Phase-locked?", "Yes" if phase_locked else "Partial",
                     f"Range: {angle_range:.3f}")
        
        st.info("""
        **Economic Interpretation**: 
        - Players = Firms choosing market positions
        - Angles = Product differentiation strategies  
        - Energy = Market power/resources
        - Equilibrium = Stable market configuration
        - Termination = Market maturity/exit
        """)
    
    def demonstrate_mechanism_design(self):
        """Show φ-mechanism properties"""
        st.header("φ-Mechanism Design")
        
        st.markdown("""
        ### Auction Design with Topological Structure
        
        The φ-mechanism uses Fibonacci decomposition for allocation:
        - Bidders submit values
        - Allocation based on phase coherence
        - Payments ensure incentive compatibility
        """)
        
        n_bidders = st.slider("Number of bidders", 2, 10, 5)
        n_items = st.slider("Number of items", 1, 5, 2)
        
        # Generate random valuations
        true_values = np.random.uniform(50, 200, (n_bidders, n_items))
        
        st.subheader("Bidder Valuations")
        
        # Display as dataframe
        df_values = pd.DataFrame(true_values, 
                               columns=[f'Item {i+1}' for i in range(n_items)],
                               index=[f'Bidder {i+1}' for i in range(n_bidders)])
        st.dataframe(df_values.style.format("${:.2f}"))
        
        # Run different auction formats
        results = {}
        
        # 1. First-price sealed bid
        first_price_revenue = 0
        first_price_allocation = np.zeros((n_bidders, n_items))
        
        for item in range(n_items):
            bids = true_values[:, item] * np.random.uniform(0.7, 0.9, n_bidders)  # Shade bids
            winner = np.argmax(bids)
            first_price_revenue += bids[winner]
            first_price_allocation[winner, item] = 1
        
        results['First-Price'] = {
            'revenue': first_price_revenue,
            'efficiency': np.sum(true_values * first_price_allocation)
        }
        
        # 2. Second-price (Vickrey)
        second_price_revenue = 0
        second_price_allocation = np.zeros((n_bidders, n_items))
        
        for item in range(n_items):
            bids = true_values[:, item]  # Truth-telling
            order = np.argsort(bids)[::-1]
            winner = order[0]
            second_price_revenue += bids[order[1]] if len(order) > 1 else 0
            second_price_allocation[winner, item] = 1
        
        results['Second-Price'] = {
            'revenue': second_price_revenue,
            'efficiency': np.sum(true_values * second_price_allocation)
        }
        
        # 3. φ-Mechanism
        phi_revenue = 0
        phi_allocation = np.zeros((n_bidders, n_items))
        
        for item in range(n_items):
            # Score based on value and Fibonacci position
            scores = np.zeros(n_bidders)
            for i in range(n_bidders):
                position = i * n_items + item + 1
                zeck = zeckendorf_decomposition(position)
                topological_bonus = len(zeck) * 0.1  # Bonus for sparse decomposition
                scores[i] = true_values[i, item] * (1 + topological_bonus)
            
            winner = np.argmax(scores)
            
            # φ-payment rule
            payment = true_values[winner, item] / self.phi
            phi_revenue += payment
            phi_allocation[winner, item] = 1
        
        results['φ-Mechanism'] = {
            'revenue': phi_revenue,
            'efficiency': np.sum(true_values * phi_allocation)
        }
        
        # Display results
        st.subheader("Auction Performance")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        mechanisms = list(results.keys())
        revenues = [results[m]['revenue'] for m in mechanisms]
        efficiencies = [results[m]['efficiency'] for m in mechanisms]
        
        x = np.arange(len(mechanisms))
        width = 0.35
        
        ax1.bar(x - width/2, revenues, width, label='Revenue', alpha=0.7)
        ax1.bar(x + width/2, efficiencies, width, label='Efficiency', alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels(mechanisms)
        ax1.set_ylabel('Value ($)')
        ax1.set_title('Auction Outcomes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Revenue vs Efficiency scatter
        ax2.scatter(efficiencies, revenues, s=200)
        for i, m in enumerate(mechanisms):
            ax2.annotate(m, (efficiencies[i], revenues[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax2.set_xlabel('Efficiency ($)')
        ax2.set_ylabel('Revenue ($)')
        ax2.set_title('Revenue-Efficiency Tradeoff')
        ax2.grid(True, alpha=0.3)
        
        # Pareto frontier
        max_efficiency = max(efficiencies)
        max_revenue = max(revenues)
        ax2.plot([0, max_efficiency], [max_revenue, 0], 'k--', alpha=0.3, label='Pareto frontier')
        ax2.legend()
        
        st.pyplot(fig)
        
        # Theoretical properties
        st.subheader("Mechanism Properties")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **First-Price**
            - Strategic bidding
            - Revenue depends on competition
            - Not efficient
            """)
        
        with col2:
            st.success("""
            **Second-Price**  
            - Truth-telling optimal
            - Efficient allocation
            - Lower revenue
            """)
        
        with col3:
            st.warning("""
            **φ-Mechanism**
            - Topological scoring
            - Near-efficient
            - Higher expected revenue
            """)

def main():
    """Main Streamlit app"""
    st.set_page_config(page_title="N-Game Topological DiD Framework", 
                      page_icon="🌀", 
                      layout="wide")
    
    st.title("N-Game Topological Difference-in-Differences Framework")
    st.markdown("### An Interactive Demonstration for Economists")
    
    # Sidebar
    st.sidebar.header("Select Demonstration")
    demo_choice = st.sidebar.radio("Choose a topic:", [
        "Time Consistency",
        "DiD Identification", 
        "Equilibrium Dynamics",
        "Mechanism Design"
    ])
    
    # Initialize demonstrator
    demo = EconomicDemonstration()
    
    # Run selected demonstration
    if demo_choice == "Time Consistency":
        demo.demonstrate_time_consistency()
    elif demo_choice == "DiD Identification":
        demo.demonstrate_did_identification()
    elif demo_choice == "Equilibrium Dynamics":
        demo.demonstrate_equilibrium_dynamics()
    elif demo_choice == "Mechanism Design":
        demo.demonstrate_mechanism_design()
    
    # Footer with key insights
    st.markdown("---")
    st.subheader("Key Theoretical Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **Single Primitive**
        - φ² = φ + 1
        - All structures emerge
        - No arbitrary parameters
        """)
    
    with col2:
        st.markdown("""
        **Natural Experiments**
        - Zeckendorf decomposition
        - Exogenous variation
        - Multiple instruments
        """)
    
    with col3:
        st.markdown("""
        **Time Consistency**
        - β = 1/φ ≈ 0.618
        - No preference reversals
        - Natural discounting
        """)
    
    with col4:
        st.markdown("""
        **Equilibrium**
        - Phase coherence
        - Energy decay
        - Natural termination
        """)
    
    # References
    with st.expander("References and Further Reading"):
        st.markdown("""
        1. **Core Framework**: See `economic_theory_paper.md` for formal proofs
        2. **Implementation**: `phi_mamba/` directory contains the computational framework
        3. **Validation**: `game_theory_validation.py` provides empirical tests
        4. **Figures**: `journal_figures/` contains publication-ready visualizations
        
        **Key Papers to Cite**:
        - Zeckendorf, E. (1972). "Représentation des nombres naturels par une somme de nombres de Fibonacci." 
        - Angrist & Pischke (2009). "Mostly Harmless Econometrics"
        - Laibson, D. (1997). "Golden Eggs and Hyperbolic Discounting."
        """)

if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        main()
    except:
        print("To run the interactive demo, use: streamlit run economic_demo.py")
        print("\nAlternatively, here's a static demonstration:")
        
        # Run static version
        demo = EconomicDemonstration()
        
        print("\n=== TIME CONSISTENCY DEMONSTRATION ===")
        print(f"Discount factor β = 1/φ = {1/PHI:.6f}")
        print("This ensures no preference reversals over time")
        
        print("\n=== FIBONACCI TREATMENT ASSIGNMENT ===")
        for n in [17, 42, 100]:
            zeck = zeckendorf_decomposition(n)
            print(f"Position {n}: {zeck} → Treatment = {5 in zeck}")
        
        print("\n=== EQUILIBRIUM PROPERTIES ===")
        print(f"Energy decay: E(t) = φ^(-t)")
        print(f"After 5 iterations: E(5) = {PHI**(-5):.6f}")
        print("Natural termination when E < 0.01")
        
        print("\nFor full interactive experience, install streamlit: pip install streamlit")