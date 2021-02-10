import os
import torch
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from neurodiffeq.monitors import MonitorSphericalHarmonics, BaseMonitor
from neurodiffeq.callbacks import MonitorCallback


class ResidualMonitorSphericalHarmonics(MonitorSphericalHarmonics):
    def __init__(self, *args, **kwargs):
        self.pde_system = kwargs.pop('pde_system')
        super(ResidualMonitorSphericalHarmonics, self).__init__(*args, **kwargs)

    def _compute_us(self, nets, conditions):
        r, theta, phi = self.r_tensor, self.theta_tensor, self.phi_tensor
        Rs = [cond.enforce(net, r) for net, cond in zip(nets, conditions)]
        fs = self.pde_system(*Rs, r, theta, phi)
        return [f.detach().cpu().numpy() for f in fs]

    @staticmethod
    def _update_r_plot_grouped_by_phi(var_name, ax, df):
        ax.clear()
        sns.lineplot(x='$r$', y='u', hue='$\\phi$', data=df, ax=ax)
        ax.set_title(f'{var_name}($r$) grouped by $\\phi$')
        ax.set_ylabel(var_name)
        ax.set_yscale('symlog')

    @staticmethod
    def _update_r_plot_grouped_by_theta(var_name, ax, df):
        ax.clear()
        sns.lineplot(x='$r$', y='u', hue='$\\theta$', data=df, ax=ax)
        ax.set_title(f'{var_name}($r$) grouped by $\\theta$')
        ax.set_ylabel(var_name)
        ax.set_yscale('symlog')


class MonitorAxisymmetricSphericalVectorField(BaseMonitor):
    def __init__(self, r_min, r_max, theta_min, theta_max, harmonics_fn, check_every,
                 density=3.0, net_r_index=0, net_theta_index=1):
        super(MonitorAxisymmetricSphericalVectorField, self).__init__()
        self.using_non_gui_backend = (matplotlib.get_backend() == 'agg')
        self.density = density
        nx = int(density * 30)
        ny = int(density * 60)
        self.r_min, self.r_max = r_min, r_max
        self.theta_min, self.theta_max = theta_min, theta_max
        self.harmonics_fn = harmonics_fn
        self.check_every = check_every
        self.net_r_index, self.net_theta_index = net_r_index, net_theta_index

        x = np.linspace(0, r_max, nx)
        y = np.linspace(-r_max, r_max, ny)
        x, y = np.meshgrid(x, y)
        self.x, self.y = x, y

        r = torch.tensor(np.sqrt(x ** 2 + y ** 2))
        theta = torch.tensor(np.pi / 2 - np.arctan2(y, x))
        phi = torch.zeros_like(theta)
        self.mesh_shape = r.shape
        mask = torch.ones_like(r)
        mask[(r_min > r) | (r > r_max) | (theta_min > theta) | (theta > theta_max)] = torch.tensor(np.nan)

        self.r = r.reshape(-1, 1)
        self.basis = harmonics_fn(theta.reshape(-1, 1), phi.reshape(-1, 1)) * mask.reshape(-1, 1)
        self.fig, self.ax, self.colorbar = None, None, None

        # plot the boundaries
        n_points_boundary_seg = 200
        th0 = np.ones(n_points_boundary_seg) * theta_min
        th1 = np.ones(n_points_boundary_seg) * theta_max
        th_forward = np.linspace(theta_min, theta_max, n_points_boundary_seg)
        th_backward = th_forward[::-1]
        r0 = np.ones(n_points_boundary_seg) * r_min
        r1 = np.ones(n_points_boundary_seg) * r_max
        r_forward = np.linspace(r_min, r_max, n_points_boundary_seg)
        r_backword = r_forward[::-1]

        r_boundary = np.concatenate([r_forward, r1, r_backword, r0])
        th_boundary = np.concatenate([th0, th_forward, th1, th_backward])

        self.x_boundary = r_boundary * np.sin(th_boundary)
        self.y_boundary = r_boundary * np.cos(th_boundary)

    def check(self, nets, conditions, history):
        if not self.fig:
            self.fig = plt.figure(figsize=(6, 10))
            self.fig.tight_layout()
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.ax.set_aspect(aspect='equal')

        net_r = nets[self.net_r_index]
        cond_r = conditions[self.net_r_index]
        net_theta = nets[self.net_theta_index]
        cond_theta = conditions[self.net_theta_index]

        with torch.no_grad():
            ur = (cond_r.enforce(net_r, self.r) * self.basis).sum(dim=1, keepdim=True)
            utheta = (cond_theta.enforce(net_theta, self.r) * self.basis).sum(dim=1, keepdim=True)
            ux = (ur * torch.sin(utheta)).reshape(self.mesh_shape).cpu().numpy()
            uy = (ur * torch.cos(utheta)).reshape(self.mesh_shape).cpu().numpy()
            color = ur.reshape(self.mesh_shape).cpu().numpy()

        self.ax.clear()
        cax = self.ax.streamplot(
            self.x, self.y, ux, uy,
            density=(self.density, 2 * self.density),
            color=color,
            cmap='magma',
        )
        self.ax.plot(self.x_boundary, self.y_boundary, linewidth=3.0, linestyle='--', color='m')
        if self.colorbar:
            self.colorbar.remove()
        self.colorbar = self.fig.colorbar(cax.lines, ax=self.ax)

        self.fig.canvas.draw()

        if self.using_non_gui_backend:
            plt.pause(0.05)


class MonitorCallbackFactory:
    monitors = {
        'solution': MonitorSphericalHarmonics,
        'residual': ResidualMonitorSphericalHarmonics,
        'axis_sym': MonitorAxisymmetricSphericalVectorField,
    }

    @staticmethod
    def from_config(cfg, pde_system, harmonics_fn, logger=None):
        callbacks = []
        for k, c in cfg.monitor.items():
            MonitorClass = MonitorCallbackFactory.monitors.get(k)
            if not k:
                raise ValueError(f"Unknown Monitor type {c}")
            kwargs = {kw: arg for kw, arg in c.items()}
            kwargs['harmonics_fn'] = harmonics_fn
            if MonitorClass == ResidualMonitorSphericalHarmonics:
                kwargs['pde_system'] = pde_system
            fig_dir = kwargs.pop('fig_dir', None)
            if fig_dir:
                fig_dir = os.path.join(cfg.meta.base_path, fig_dir)
            callbacks.append(MonitorCallback(MonitorClass(**kwargs), fig_dir=fig_dir, logger=logger))

        return callbacks
