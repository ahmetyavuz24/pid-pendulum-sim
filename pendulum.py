import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

class PhysicalPendulum:
    """Physical pendulum with proper 0=up coordinates"""
    def __init__(self, L=1.0, m=1.0, g=9.81, dt=0.01):
        self.L = L
        self.m = m
        self.g = g
        self.dt = dt
        self.θ = np.pi
        self.ω = 0.0

    def step(self, τ):
        """Update pendulum state using torque input"""
        α = (τ - self.m*self.g*self.L*np.sin(self.θ))/(self.m*self.L**2)
        self.ω += α * self.dt
        self.θ += self.ω * self.dt
        self.θ = (self.θ + np.pi) % (2*np.pi) - np.pi
        return self.θ, self.ω

class PIDController:
    """PID controller for pendulum"""
    def __init__(self, Kp, Ki, Kd, dt, name="PID"):
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki)
        self.Kd = np.array(Kd)
        self.dt = dt
        self.integral = 0
        self.prev_error = 0
        self.name = name
        self.Kp_history = []
        self.Ki_history = []
        self.Kd_history = []

    def control(self, θ, ω, target, t_index):
        """Control with potentially time-varying PID parameters"""
        Kp = self.Kp[t_index] if t_index < len(self.Kp) else self.Kp[-1]
        Ki = self.Ki[t_index] if t_index < len(self.Ki) else self.Ki[-1]
        Kd = self.Kd[t_index] if t_index < len(self.Kd) else self.Kd[-1]

        error = (target - θ + np.pi) % (2*np.pi) - np.pi
        self.integral += error * self.dt
        derivative = (error - self.prev_error)/self.dt
        self.prev_error = error

        self.Kp_history.append(Kp)
        self.Ki_history.append(Ki)
        self.Kd_history.append(Kd)

        return Kp*error + Ki*self.integral + Kd*derivative

def animate_single_varying_pid(pendulum, pid_params_over_time, target_angle, duration):
    """Animate a single controller with PID parameters changing over time"""
    dt = pendulum.dt
    t = np.arange(0, duration, dt)
    θ_vals, ω_vals, τ_vals = [], [], []

    # Initialize PID controller with the first set of parameters
    initial_params = pid_params_over_time[0]
    pid_controller = PIDController(
        [p['Kp'] for p in pid_params_over_time],
        [p['Ki'] for p in pid_params_over_time],
        [p['Kd'] for p in pid_params_over_time],
        dt,
        name="Varying PID"
    )

    for i in range(len(t)):
        θ, ω = pendulum.step(τ_vals[-1] if τ_vals else 0)
        τ = pid_controller.control(θ, ω, target_angle, i)  # Pass the current time index
        θ_vals.append(θ)
        ω_vals.append(ω)
        τ_vals.append(τ)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Pendulum animation setup
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_title(f'Pendulum Animation: {pid_controller.name}')

    line, = ax1.plot([], [], 'k-', lw=2)
    bob = Circle((0, 0), 0.1, fc='r')
    ax1.add_patch(bob)
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
    target_line = ax1.plot([0, np.sin(target_angle)], [0, np.cos(target_angle)],
                          'r--', alpha=0.3)[0]

    # Angle plot
    ax2.set_xlim(0, duration)
    ax2.set_ylim(-np.pi, np.pi)
    ax2.axhline(target_angle, color='r', linestyle='--', alpha=0.3, label='Target')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (rad)')
    ax2.grid(True)
    ax2.set_title('Angle Response')
    angle_line, = ax2.plot([], [], 'b-', label='Actual')
    ax2.legend()

    # PID parameters plot
    ax3.set_xlim(0, duration)
    # Set y-axis limits based on the range of PID parameters
    all_kp = [p['Kp'] for p in pid_params_over_time]
    all_ki = [p['Ki'] for p in pid_params_over_time]
    all_kd = [p['Kd'] for p in pid_params_over_time]
    ax3.set_ylim(0, max(max(all_kp), max(all_ki), max(all_kd)) * 1.1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Gain Value')
    ax3.set_title('PID Parameters vs Time')
    ax3.grid(True)
    kp_line, = ax3.plot([], [], 'r-', label='Kp')
    ki_line, = ax3.plot([], [], 'g--', label='Ki')
    kd_line, = ax3.plot([], [], 'b:', label='Kd')
    ax3.legend()

    def init():
        line.set_data([], [])
        bob.center = (0, 0)
        time_text.set_text('')
        angle_line.set_data([], [])
        kp_line.set_data([], [])
        ki_line.set_data([], [])
        kd_line.set_data([], [])
        return line, bob, time_text, angle_line, kp_line, ki_line, kd_line

    def update(frame):
        x = [0, np.sin(θ_vals[frame])]
        y = [0, np.cos(θ_vals[frame])]
        line.set_data(x, y)
        bob.center = (x[1], y[1])
        time_text.set_text(f'Time: {t[frame]:.2f}s\nAngle: {θ_vals[frame]:.2f}rad')
        angle_line.set_data(t[:frame+1], θ_vals[:frame+1])
        kp_line.set_data(t[:frame+1], pid_controller.Kp_history[:frame+1])
        ki_line.set_data(t[:frame+1], pid_controller.Ki_history[:frame+1])
        kd_line.set_data(t[:frame+1], pid_controller.Kd_history[:frame+1])
        return line, bob, time_text, angle_line, kp_line, ki_line, kd_line

    ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=20)
    plt.tight_layout()
    plt.show()
    return ani

# Example usage remains the same
if __name__ == "__main__":
    # Define pendulum parameters
    pendulum_params = {'L': 1.0, 'm': 1.0, 'g': 9.81, 'dt': 0.01}
    duration = 10  # seconds
    dt = pendulum_params['dt']
    num_steps = int(duration / dt)

    # Define PID parameters for each step (constant for now)
    pid_params_over_time = []
    kp_val = 14.65
    ki_val = 6.45
    kd_val = 3.08
    for _ in range(num_steps):
        pid_params_over_time.append({
            'Kp': kp_val,
            'Ki': ki_val,
            'Kd': kd_val
        })

    # Target angle (45 degrees)
    target_angle = 45 * np.pi/180

    # Animate with varying PID parameters
    pendulum = PhysicalPendulum(**pendulum_params)
    ani = animate_single_varying_pid(pendulum, pid_params_over_time, target_angle, duration)