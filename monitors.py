import os
import seaborn as sns
from neurodiffeq.monitors import MonitorSphericalHarmonics
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


class MonitorCallbackFactory:
    monitors = {
        'solution': MonitorSphericalHarmonics,
        'residual': ResidualMonitorSphericalHarmonics,
    }

    @staticmethod
    def from_config(cfg, pde_system, harmonics_fn):
        callbacks = []
        for k, c in cfg.items():
            MonitorClass = MonitorCallbackFactory.monitors.get(k)
            if not k:
                raise ValueError(f"Unknown Monitor type {c}")
            kwargs = {kw: arg for kw, arg in c.items()}
            kwargs['harmonics_fn'] = harmonics_fn
            if MonitorClass == ResidualMonitorSphericalHarmonics:
                kwargs['pde_system'] = pde_system
            fig_dir = kwargs.pop('fig_dir', None)
            callbacks.append(MonitorCallback(MonitorClass(**kwargs), fig_dir=fig_dir))

        return callbacks
