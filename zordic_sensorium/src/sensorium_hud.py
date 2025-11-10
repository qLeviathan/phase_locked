"""
ZORDIC SENSORIUM - AI HUD Interface
Real-time visualization of AI internal state

Architecture:
- Real-time φ-field dynamics visualization
- Token processing stream
- Regime state monitoring (deterministic/stochastic)
- Network activity (DID when integrated)
- Memory/computation metrics
- Visual feedback of AI "awareness"
"""

import pygame
import numpy as np
from collections import deque
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
FPS = 60

# Colors (cyberpunk aesthetic)
COLOR_BG = (5, 8, 15)
COLOR_PRIMARY = (0, 255, 170)  # Cyan
COLOR_SECONDARY = (255, 60, 100)  # Pink
COLOR_WARNING = (255, 170, 0)  # Orange
COLOR_TEXT = (200, 220, 240)
COLOR_GRID = (20, 30, 40)

PHI = (1 + np.sqrt(5)) / 2
PSI = (1 - np.sqrt(5)) / 2


@dataclass
class SensorReading:
    """Single sensor data point"""
    timestamp: float
    value: float
    label: str
    color: Tuple[int, int, int]


class DataStream:
    """Real-time data stream with history"""

    def __init__(self, max_length: int = 200):
        self.data = deque(maxlen=max_length)
        self.max_length = max_length

    def add(self, value: float):
        self.data.append(value)

    def get_normalized(self) -> List[float]:
        """Get data normalized to 0-1 range"""
        if not self.data:
            return []

        data_array = np.array(self.data)
        min_val = np.min(data_array)
        max_val = np.max(data_array)

        if max_val - min_val < 1e-6:
            return [0.5] * len(self.data)

        return ((data_array - min_val) / (max_val - min_val)).tolist()

    def get_raw(self) -> List[float]:
        return list(self.data)


class PhiFieldVisualizer:
    """Visualizes φ/ψ conjugate field dynamics"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)

        # Field state
        self.phi_stream = DataStream(max_length=100)
        self.psi_stream = DataStream(max_length=100)
        self.delta_stream = DataStream(max_length=100)

    def update(self, phi: float, psi: float):
        """Update field state"""
        self.phi_stream.add(phi)
        self.psi_stream.add(psi)
        self.delta_stream.add(abs(phi - psi))

    def render(self) -> pygame.Surface:
        """Render φ-field visualization"""
        self.surface.fill((0, 0, 0, 0))

        # Draw φ wave (cyan)
        phi_data = self.phi_stream.get_normalized()
        if len(phi_data) > 1:
            points = [(i * self.width / len(phi_data),
                      self.height - phi_data[i] * self.height * 0.8 - 20)
                     for i in range(len(phi_data))]
            pygame.draw.lines(self.surface, COLOR_PRIMARY, False, points, 2)

        # Draw ψ wave (pink)
        psi_data = self.psi_stream.get_normalized()
        if len(psi_data) > 1:
            points = [(i * self.width / len(psi_data),
                      self.height - psi_data[i] * self.height * 0.8 - 20)
                     for i in range(len(psi_data))]
            pygame.draw.lines(self.surface, COLOR_SECONDARY, False, points, 2)

        # Draw delta stability indicator
        delta_data = self.delta_stream.get_normalized()
        if len(delta_data) > 1:
            points = [(i * self.width / len(delta_data),
                      self.height - delta_data[i] * self.height * 0.4 - 10)
                     for i in range(len(delta_data))]
            pygame.draw.lines(self.surface, COLOR_WARNING, False, points, 1)

        # Labels
        font = pygame.font.Font(None, 24)
        phi_label = font.render("φ-field", True, COLOR_PRIMARY)
        psi_label = font.render("ψ-field", True, COLOR_SECONDARY)
        self.surface.blit(phi_label, (10, 10))
        self.surface.blit(psi_label, (10, 35))

        return self.surface


class TokenStreamVisualizer:
    """Visualizes token processing stream"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)

        self.tokens = deque(maxlen=50)
        self.token_positions = deque(maxlen=50)

    def add_token(self, char: str, phi: float, psi: float, stable: bool):
        """Add token to stream"""
        self.tokens.append({
            'char': char,
            'phi': phi,
            'psi': psi,
            'stable': stable,
            'time': time.time()
        })

    def update(self):
        """Update token positions (scrolling effect)"""
        current_time = time.time()

        # Remove old tokens
        while self.tokens and current_time - self.tokens[0]['time'] > 5.0:
            self.tokens.popleft()

    def render(self) -> pygame.Surface:
        """Render token stream"""
        self.surface.fill((0, 0, 0, 0))

        font = pygame.font.Font(None, 32)
        current_time = time.time()

        for i, token in enumerate(self.tokens):
            age = current_time - token['time']
            alpha = max(0, 255 - int(age * 50))

            # Color based on stability
            if token['stable']:
                color = (*COLOR_PRIMARY, alpha)
            else:
                color = (*COLOR_WARNING, alpha)

            # Position
            x = self.width - 50 - i * 30
            y = self.height // 2

            # Draw token
            text = font.render(token['char'], True, color)

            # Glow effect for unstable
            if not token['stable']:
                glow_surface = pygame.Surface((40, 40), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (*COLOR_WARNING, alpha // 3), (20, 20), 20)
                self.surface.blit(glow_surface, (x - 10, y - 10))

            self.surface.blit(text, (x, y))

        return self.surface


class RegimeIndicator:
    """Shows current regime state"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)

        self.regime = "MIXED"
        self.deterministic_ratio = 0.5
        self.history = DataStream(max_length=100)

    def update(self, regime: str, ratio: float):
        """Update regime state"""
        self.regime = regime
        self.deterministic_ratio = ratio
        self.history.add(ratio)

    def render(self) -> pygame.Surface:
        """Render regime indicator"""
        self.surface.fill((0, 0, 0, 0))

        # Title
        font_title = pygame.font.Font(None, 28)
        title = font_title.render("REGIME STATE", True, COLOR_TEXT)
        self.surface.blit(title, (10, 10))

        # Current regime (large)
        font_regime = pygame.font.Font(None, 48)

        if self.regime == "DETERMINISTIC":
            color = COLOR_PRIMARY
        elif self.regime == "STOCHASTIC":
            color = COLOR_SECONDARY
        else:
            color = COLOR_WARNING

        regime_text = font_regime.render(self.regime, True, color)
        self.surface.blit(regime_text, (10, 40))

        # Ratio bar
        bar_width = self.width - 20
        bar_height = 30
        bar_x = 10
        bar_y = 100

        # Background
        pygame.draw.rect(self.surface, (30, 40, 50),
                        (bar_x, bar_y, bar_width, bar_height))

        # Fill
        fill_width = int(bar_width * self.deterministic_ratio)
        pygame.draw.rect(self.surface, color,
                        (bar_x, bar_y, fill_width, bar_height))

        # Border
        pygame.draw.rect(self.surface, COLOR_TEXT,
                        (bar_x, bar_y, bar_width, bar_height), 2)

        # Ratio text
        font_small = pygame.font.Font(None, 24)
        ratio_text = font_small.render(f"{self.deterministic_ratio:.1%}", True, COLOR_TEXT)
        self.surface.blit(ratio_text, (bar_x + bar_width + 10, bar_y + 5))

        # History graph
        if len(self.history.data) > 1:
            graph_y = bar_y + 50
            graph_height = 60

            points = [(i * bar_width / len(self.history.data),
                      graph_y + graph_height - self.history.data[i] * graph_height)
                     for i in range(len(self.history.data))]

            pygame.draw.lines(self.surface, color, False, points, 2)

        return self.surface


class MetricsPanel:
    """Shows system metrics"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)

        self.metrics = {}

    def update_metric(self, key: str, value: float, unit: str = ""):
        """Update metric value"""
        self.metrics[key] = {'value': value, 'unit': unit}

    def render(self) -> pygame.Surface:
        """Render metrics panel"""
        self.surface.fill((0, 0, 0, 0))

        font = pygame.font.Font(None, 24)
        y_offset = 10

        for key, data in self.metrics.items():
            label = font.render(f"{key}:", True, COLOR_TEXT)
            value = font.render(f"{data['value']:.3f} {data['unit']}", True, COLOR_PRIMARY)

            self.surface.blit(label, (10, y_offset))
            self.surface.blit(value, (150, y_offset))

            y_offset += 30

        return self.surface


class ZordicSensorium:
    """Main AI HUD/Sensorium interface"""

    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("ZORDIC SENSORIUM - AI HUD")
        self.clock = pygame.time.Clock()
        self.running = True

        # Components
        self.phi_field = PhiFieldVisualizer(800, 300)
        self.token_stream = TokenStreamVisualizer(800, 200)
        self.regime_indicator = RegimeIndicator(400, 200)
        self.metrics_panel = MetricsPanel(300, 400)

        # Data sources (simulation for now)
        self.simulation_thread = None
        self.data_lock = threading.Lock()

        # Current state
        self.current_phi = 0.0
        self.current_psi = 0.0
        self.current_regime = "MIXED"
        self.current_ratio = 0.5

    def start_simulation(self):
        """Start simulated data stream"""
        def simulate():
            t = 0
            while self.running:
                # Simulate φ/ψ field oscillation
                phi = abs(np.sin(t * 0.1) * PHI)
                psi = abs(np.cos(t * 0.15) * abs(PSI))

                with self.data_lock:
                    self.current_phi = phi
                    self.current_psi = psi

                    # Regime based on delta
                    delta = abs(phi - psi)
                    if delta < 0.5:
                        self.current_regime = "DETERMINISTIC"
                        self.current_ratio = 0.9
                    elif delta > 1.5:
                        self.current_regime = "STOCHASTIC"
                        self.current_ratio = 0.2
                    else:
                        self.current_regime = "MIXED"
                        self.current_ratio = 0.5

                    # Occasional token
                    if np.random.random() < 0.1:
                        char = np.random.choice(list("abcdefghijklmnopqrstuvwxyz "))
                        stable = delta < 0.8
                        self.token_stream.add_token(char, phi, psi, stable)

                t += 1
                time.sleep(0.1)

        self.simulation_thread = threading.Thread(target=simulate, daemon=True)
        self.simulation_thread.start()

    def draw_grid(self):
        """Draw background grid"""
        # Horizontal lines
        for y in range(0, SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, COLOR_GRID, (0, y), (SCREEN_WIDTH, y), 1)

        # Vertical lines
        for x in range(0, SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, COLOR_GRID, (x, 0), (x, SCREEN_HEIGHT), 1)

    def draw_hud_frame(self):
        """Draw HUD frame elements"""
        # Corner brackets
        bracket_size = 30
        bracket_thickness = 3

        corners = [
            (20, 20),  # Top-left
            (SCREEN_WIDTH - 20, 20),  # Top-right
            (20, SCREEN_HEIGHT - 20),  # Bottom-left
            (SCREEN_WIDTH - 20, SCREEN_HEIGHT - 20)  # Bottom-right
        ]

        for x, y in corners:
            # Top-left corner style
            if x < SCREEN_WIDTH // 2 and y < SCREEN_HEIGHT // 2:
                pygame.draw.line(self.screen, COLOR_PRIMARY, (x, y), (x + bracket_size, y), bracket_thickness)
                pygame.draw.line(self.screen, COLOR_PRIMARY, (x, y), (x, y + bracket_size), bracket_thickness)
            # Top-right
            elif x > SCREEN_WIDTH // 2 and y < SCREEN_HEIGHT // 2:
                pygame.draw.line(self.screen, COLOR_PRIMARY, (x, y), (x - bracket_size, y), bracket_thickness)
                pygame.draw.line(self.screen, COLOR_PRIMARY, (x, y), (x, y + bracket_size), bracket_thickness)
            # Bottom-left
            elif x < SCREEN_WIDTH // 2 and y > SCREEN_HEIGHT // 2:
                pygame.draw.line(self.screen, COLOR_PRIMARY, (x, y), (x + bracket_size, y), bracket_thickness)
                pygame.draw.line(self.screen, COLOR_PRIMARY, (x, y), (x, y - bracket_size), bracket_thickness)
            # Bottom-right
            else:
                pygame.draw.line(self.screen, COLOR_PRIMARY, (x, y), (x - bracket_size, y), bracket_thickness)
                pygame.draw.line(self.screen, COLOR_PRIMARY, (x, y), (x, y - bracket_size), bracket_thickness)

        # Title
        font_title = pygame.font.Font(None, 56)
        title = font_title.render("ZORDIC SENSORIUM", True, COLOR_PRIMARY)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 40))
        self.screen.blit(title, title_rect)

        # Subtitle
        font_subtitle = pygame.font.Font(None, 28)
        subtitle = font_subtitle.render("AI State Monitor - φ-Field Dynamics", True, COLOR_TEXT)
        subtitle_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, 75))
        self.screen.blit(subtitle, subtitle_rect)

    def update(self):
        """Update all components"""
        with self.data_lock:
            # Update visualizers
            self.phi_field.update(self.current_phi, self.current_psi)
            self.token_stream.update()
            self.regime_indicator.update(self.current_regime, self.current_ratio)

            # Update metrics
            self.metrics_panel.update_metric("φ-field", self.current_phi)
            self.metrics_panel.update_metric("ψ-field", self.current_psi)
            self.metrics_panel.update_metric("Δ", abs(self.current_phi - self.current_psi))
            self.metrics_panel.update_metric("FPS", self.clock.get_fps())

    def render(self):
        """Render entire HUD"""
        # Background
        self.screen.fill(COLOR_BG)
        self.draw_grid()

        # HUD frame
        self.draw_hud_frame()

        # Components
        # φ-field visualization (center-top)
        phi_surf = self.phi_field.render()
        self.screen.blit(phi_surf, (SCREEN_WIDTH // 2 - 400, 120))

        # Token stream (center-middle)
        token_surf = self.token_stream.render()
        self.screen.blit(token_surf, (SCREEN_WIDTH // 2 - 400, 450))

        # Regime indicator (right side)
        regime_surf = self.regime_indicator.render()
        self.screen.blit(regime_surf, (SCREEN_WIDTH - 450, 120))

        # Metrics panel (left side)
        metrics_surf = self.metrics_panel.render()
        self.screen.blit(metrics_surf, (50, 120))

        # Status bar (bottom)
        font = pygame.font.Font(None, 20)
        status = font.render(f"SYSTEM ONLINE | TIME: {time.time():.2f} | REGIME: {self.current_regime}",
                           True, COLOR_TEXT)
        self.screen.blit(status, (50, SCREEN_HEIGHT - 40))

        pygame.display.flip()

    def handle_events(self):
        """Handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # Toggle simulation
                    pass

    def run(self):
        """Main loop"""
        self.start_simulation()

        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FPS)

        pygame.quit()


def main():
    """Entry point"""
    print("="*70)
    print("  ZORDIC SENSORIUM - AI HUD Interface")
    print("  Real-time visualization of AI internal state")
    print("="*70)
    print()
    print("Initializing sensorium...")
    print()

    sensorium = ZordicSensorium()
    sensorium.run()


if __name__ == "__main__":
    main()
