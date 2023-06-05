import casadi as ca
import do_mpc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# Customizing Matplotlib:
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

# TODO: Tune this weights
# TODO: Make this variables a file to be read for readability
CTE_W = 500 # Cross Track Error Weight
EPSI_W = 300 # Heading Error Weight
V_W = 100 # Velocity Weight

A_W = 20 # Acceleration Weight
DELTA_W = 20 # Steering Angle Weight

A_RATE_W = 100 # Acceleration Rate Weight
DELTA_RATE_W = 60 # Steering Angle Rate Weight

DESIRED_SPEED = 20 # Desired speed

class ModelPredictiveController:
    """
    # MPC Implementation
    This class implements a Model Predictive Controller (MPC) for a vehicle in CARLA. The MPC is used to Control the steering angle and throttle of the vehicle in order to follow a reference trajectory.

    ## Equations for the model, from https://cacheop.medium.com/implementing-a-model-predictive-control-for-a-self-driving-car-7ee6212a04a8
    x_[t] = x[t-1] + v[t-1] * cos(psi[t-1]) * dt
    y_[t] = y[t-1] + v[t-1] * sin(psi[t-1]) * dt
    psi_[t] = psi[t-1] + v[t-1] / Lf * delta[t-1] * dt
    v_[t] = v[t-1] + a[t-1] * dt
    cte[t] = f(x[t-1]) - y[t-1] + v[t-1] * sin(epsi[t-1]) * dt
    epsi[t] = psi[t] - psides[t-1] + v[t-1] * delta[t-1] / Lf * dt

    ## Control inputs
    a = acceleration, [-1, 1] [full brake, full throttle]
    delta = steering angle, [-25, 25] degrees
    """
    def __init__(self, dt=0.05, N=10, L=3):
        self.dt = dt # Time step
        self.N = N # Prediction horizon
        self.L = L # Number of constraints

        self.v_des = DESIRED_SPEED # Desired speed

        self.p_deg = 3 # Degree of the polynomial
        self.coeff = [0] * (self.p_deg + 1) # Coefficients of the polynomial

        self.model = None
        self.mpc = None

        self.setup_model()
        self.setup_mpc()
        # self.setup_simulator()

    def setup_simulator(self):
        """
        Setup simulator using do-mpc simulator.
        """
        self.simulator = do_mpc.simulator.Simulator(self.model)

        setup_simulator = {
            'integration_tool': 'cvodes',
            'abstol': 1e-8,
            'reltol': 1e-8,
            't_step': self.dt,
        }

        self.simulator.set_param(**setup_simulator)

        tvp_temp = self.simulator.get_tvp_template()

        def tvp_fun(t_now):
            for i in range(self.N + 1):
                tvp_temp['coeff'] = self.coeff
            
            return tvp_temp
        
        self.simulator.set_tvp_fun(tvp_fun)

        self.simulator.setup()

    def setup_model(self):
        """
        Setup model using do-mpc model.
        """
        model_type = 'continuous' # either 'discrete' or 'continuous' model
        self.model = do_mpc.model.Model(model_type)

        # States
        x = self.model.set_variable(var_type='_x', var_name='x', shape=(1,1))
        y = self.model.set_variable(var_type='_x', var_name='y', shape=(1,1))
        psi = self.model.set_variable(var_type='_x', var_name='psi', shape=(1,1))
        v = self.model.set_variable(var_type='_x', var_name='v', shape=(1,1))
        cte = self.model.set_variable(var_type='_x', var_name='cte', shape=(1,1))
        epsi = self.model.set_variable(var_type='_x', var_name='epsi', shape=(1,1))

        # Controls
        a = self.model.set_variable(var_type='_u', var_name='a', shape=(1,1))
        delta = self.model.set_variable(var_type='_u', var_name='delta', shape=(1,1))

        # Time-varying parameters
        coeff = self.model.set_variable(var_type='_tvp', var_name='coeff', shape=(self.p_deg + 1, 1))
        v_des = self.model.set_variable(var_type='_p', var_name='v_des', shape=(1,1))

        # Model equations
        curve = ca.polyval(coeff, x)
        # Use derivative of polynomial to get heading angle
        psides = ca.arctan(sum([coeff[-(i + 1)] * i*x**(i - 1) for i in range(1, self.p_deg + 1)]))

        x_next = x + v * ca.cos(psi) * self.dt
        y_next = y + v * ca.sin(psi) * self.dt
        psi_next = psi + v * delta / self.L * self.dt
        v_next = v + a * self.dt
        cte_next = curve - y + v * ca.sin(epsi) * self.dt
        epsi_next = psi - psides + (v / self.L) * delta * self.dt

        self.model.set_rhs('x', x_next)
        self.model.set_rhs('y', y_next)
        self.model.set_rhs('psi', psi_next)
        self.model.set_rhs('v', v_next)
        self.model.set_rhs('cte', cte_next)
        self.model.set_rhs('epsi', epsi_next)

        assert type(x_next) == ca.SX and type(y_next) == ca.SX and type(psi_next) == ca.SX and type(v_next) == ca.SX and type(cte_next) == ca.SX and type(epsi_next) == ca.SX, "Model equations must be of type SX"

        self.model.setup()

    def setting_graphics(self):
        self.mpc_graphics = do_mpc.graphics.Graphics(self.mpc.data)
        self.sim_graphics = do_mpc.graphics.Graphics(self.simulator.data)

        self.fig, ax = plt.subplots(3, sharex=True, figsize=(16,9))
        self.fig.align_ylabels()

        for g in [self.mpc_graphics, self.sim_graphics]:
            g.add_line(var_type='_x', var_name='psi', axis=ax[0])
            g.add_line(var_type='_x', var_name='v', axis=ax[1])

            g.add_line(var_type='_u', var_name='delta', axis=ax[2])
            g.add_line(var_type='_u', var_name='a', axis=ax[2])

    def setup_mpc(self):
        """
        Setup MPC using do-mpc controller.
        """
        self.mpc = do_mpc.controller.MPC(self.model)

        setup_mpc = {
            'n_horizon': self.N,
            't_step': self.dt,
            'n_robust': 0,
            'store_full_solution': True,
        }

        self.mpc.set_param(**setup_mpc)
        self.mpc.settings.supress_ipopt_output()

        # Get the model variables
        x = self.model.x['x']
        y = self.model.x['y']
        psi = self.model.x['psi']
        v = self.model.x['v']
        cte = self.model.x['cte']
        epsi = self.model.x['epsi']

        # Get the control variables
        a = self.model.u['a']
        delta = self.model.u['delta']

        # Desired speed
        v_des = self.model.p['v_des']

        # Set the objective function
        lterm = CTE_W * cte**2 + EPSI_W * epsi**2 + V_W * (v - v_des)**2 \
                + A_W * a**2 + DELTA_W * delta**2
        
        mterm = ca.SX(0)

        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        self.mpc.set_rterm(
            delta=DELTA_RATE_W,
            a=A_RATE_W
        )

        # Set constraints
        self.mpc.bounds['lower', '_u', 'a'] = -1
        self.mpc.bounds['upper', '_u', 'a'] = 1

        self.mpc.bounds['lower', '_u', 'delta'] = -1
        self.mpc.bounds['upper', '_u', 'delta'] = 1

        tvp_temp = self.mpc.get_tvp_template()
        p_temp = self.mpc.get_p_template(1)

        def tvp_fun(t_now):
            for i in range(self.N + 1):
                tvp_temp['_tvp', i] = self.coeff
            return tvp_temp
        
        def p_fun(t_now):
            p_temp['_p', 0] = self.v_des
            return p_temp
        
        self.mpc.set_tvp_fun(tvp_fun)
        self.mpc.set_p_fun(p_fun)

        # self.mpc.scaling['_u', 'a'] = 0.5
        # self.mpc.scaling['_u', 'delta'] = 3

        self.mpc.setup()

    def set_desired_speed(self, v_des):
        """
        Set the desired speed.

        Args:
            v_des (float): Desired speed.
        """
        self.v_des = v_des

    def get_initial_state(self, x0, throttle, steer, step_time, coeff):
        """
        // Latency
        const double delay_t = 0.1;
        // Future state considering latency into
        double delay_x = v * delay_t;
        double delay_y = 0;
        double delay_psi = v * -steer_value / Lf * delay_t;
        double delay_v = v + throttle_value * delay_t;
        double delay_cte = cte + v * sin(epsi) * delay_t;
        double delay_epsi = epsi + v * -steer_value /Lf * delay_t;
        """
        latency = self.dt

        # Predict the movement long enough to cover the manuever
        look_ahead = 3 # Look ahead distance
        cte = np.polyval(coeff, look_ahead) - x0[1] # 
        yaw_err = np.arctan(sum([coeff[-(i + 1)] * i*look_ahead**(i - 1) for i in range(1, self.p_deg + 1)])) # Using x = look_ahead to get heading angle, because we want to get the heading angle at the next time step. And also the current heading angle is 0.

        delay_x = x0[3] * latency
        delay_y = np.polyval(coeff, 0)
        delay_psi = x0[3] * -steer * latency / self.L
        delay_v = x0[3] + throttle * latency
        delay_cte = cte + x0[3] * np.sin(yaw_err) * latency
        delay_epsi = yaw_err + x0[3] * -steer / self.L * latency

        return np.array([delay_x, delay_y, delay_psi, delay_v, delay_cte, delay_epsi])

    def update_coeff(self, coeff):
        """
        Update the coefficients of the polynomial.

        Args:
            coeff (list): List of polynomial coefficients.
        """
        self.coeff = coeff

    def set_init_guess(self, x0, u0 = None):
        """
        Set the initial guess for the MPC.

        Args:
            x0 (list): List of initial states.
            u0 (list): List of initial controls.
        """
        self.mpc.x0 = x0
        # self.simulator.x0 = x0

        if u0 is not None:
            self.mpc.u0 = u0

        self.mpc.set_initial_guess()

    def step(self, x0):
        """
        Step the MPC.

        Args:
            x0 (list): List of initial states.

        Returns:
            list: List of optimal controls.
        """
        u0 = self.mpc.make_step(x0)

        return u0
    
    def get_coeffs(self, waypoints):
        """
        Get the coefficients of the polynomial.

        Args:
            waypoints (list): List of waypoints.

        Returns:
            list: List of polynomial coefficients.
        """
        coeffs = np.polyfit(waypoints[0], waypoints[1], self.p_deg)

        return coeffs