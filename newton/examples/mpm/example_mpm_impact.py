# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MPM Granular Impact -- Unified Force Law
#
# Drops a rigid ball into a bed of dry, noncohesive granular media (glass
# beads) modeled with the implicit MPM solver. Two-way coupling lets the
# sand exert a stopping force on the ball, so the projectile decelerates and
# comes to rest inside the bed.
#
# This reproduces Katsuragi & Durian, "Unified force law for granular impact
# cratering", Nature Physics 3, 420 (2007). The stopping force decomposes
# into a depth-dependent Coulomb friction term plus a velocity-dependent
# inertial drag term:
#
#     F = -m g + k|z| + m v^2 / d1
#
# where z is penetration depth, v is speed, k the friction coefficient and
# d1 the (depth-independent) inertial-drag length.
#
# Default: a single ball drop into the confined bed (for viewing).
#   python -m newton.examples mpm_impact
#
# --force-law: drop the ball from several heights, record its dynamics, and
# fit the force law using the paper's Fig. 3 method (sample (v, F) at fixed
# depths across drop heights, joint fit with a shared d1). Run headless:
#   python -m newton.examples mpm_impact --force-law --viewer null
#
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM

# Gravity magnitude [m/s^2]; the impact speed is sqrt(2 * GRAVITY * drop_height).
GRAVITY = 9.81


@wp.kernel
def compute_body_forces(
    dt: float,
    collider_ids: wp.array[int],
    collider_impulses: wp.array[wp.vec3],
    collider_impulse_pos: wp.array[wp.vec3],
    body_ids: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
):
    """Convert per-node sand impulses into forces/torques on the rigid ball.

    Mirrors the two-way coupling kernel in ``example_mpm_twoway_coupling.py``.
    """

    i = wp.tid()

    cid = collider_ids[i]
    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        if body_index == -1:
            return

        f_world = collider_impulses[i] / dt

        X_wb = body_q[body_index]
        X_com = body_com[body_index]
        r = collider_impulse_pos[i] - wp.transform_point(X_wb, X_com)
        wp.atomic_add(body_f, body_index, wp.spatial_vector(f_world, wp.cross(r, f_world)))


@wp.kernel
def subtract_body_force(
    dt: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_inv_inertia: wp.array[wp.mat33],
    body_inv_mass: wp.array[float],
    body_q_res: wp.array[wp.transform],
    body_qd_res: wp.array[wp.spatial_vector],
):
    """Remove the sand-applied wrench from the body velocity seen by MPM.

    Needed so the next MPM step can recompute the complementarity-based
    frictional contact impulses from scratch (see the two-way coupling example).
    """

    body_id = wp.tid()

    f = body_f[body_id]
    delta_v = dt * body_inv_mass[body_id] * wp.spatial_top(f)
    r = wp.transform_get_rotation(body_q[body_id])
    delta_w = dt * wp.quat_rotate(r, body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f)))

    body_q_res[body_id] = body_q[body_id]
    body_qd_res[body_id] = body_qd[body_id] - wp.spatial_vector(delta_v, delta_w)


def impact_speed(drop_height: float) -> float:
    """Free-fall speed [m/s] reached after falling ``drop_height`` [m]."""
    return math.sqrt(2.0 * GRAVITY * max(drop_height, 0.0))


def _smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    """Light moving-average smoothing of the noisy per-frame coupling force."""
    if x.size < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


class Example:
    def __init__(self, viewer, args):
        # simulation timing (fine sampling to resolve the fast impact)
        self.fps = float(args.fps)
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = int(args.substeps)
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        # impactor parameters (paper knobs)
        self.ball_density = float(args.ball_density)
        self.ball_diameter = float(args.ball_diameter)
        self.drop_height = float(args.drop_height)
        self.ball_radius = 0.5 * self.ball_diameter
        self.impact_speed = impact_speed(self.drop_height)
        # start the ball this far above the surface so the descent is visible;
        # its launch velocity is reduced so it still reaches ``impact_speed`` at
        # the surface (see ``_build_ball``), keeping the physics unchanged.
        self.approach_gap = float(args.approach_gap) * self.ball_diameter

        # granular bed, sized in ball diameters so penetration fits inside it
        self.voxel_size = float(args.voxel_size)
        self.bed_half_width = args.bed_width_diameters * self.ball_diameter
        self.bed_depth = args.bed_depth_diameters * self.ball_diameter
        self.bed_surface_z = self.bed_depth
        self.sand_density = float(args.sand_density)
        self.sand_friction = float(args.friction)
        self.sand_yield_pressure = float(args.yield_pressure)
        self.sand_young_modulus = float(args.young_modulus)

        # ---- rigid-body model: a single free-falling ball + walls + ground ----
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.5
        self.ball = self._build_ball(builder)
        self._add_container(builder)
        builder.add_ground_plane()

        # ---- sand model: MPM particle bed ----
        sand_builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(sand_builder)
        self._emit_particles(sand_builder)

        self.model = builder.finalize()
        self.sand_model = sand_builder.finalize()
        self.model.set_gravity((0.0, 0.0, -GRAVITY))
        self.sand_model.set_gravity((0.0, 0.0, -GRAVITY))

        # Drucker-Prager rheology: yield pressure high enough that the overburden
        # pressure (hence Coulomb friction on the ball) grows with depth, which
        # is required to recover the depth-dependent friction term k|z|.
        self.sand_model.mpm.young_modulus.fill_(self.sand_young_modulus)
        self.sand_model.mpm.yield_pressure.fill_(self.sand_yield_pressure)
        self.sand_model.mpm.friction.fill_(self.sand_friction)

        # ---- MPM solver (sparse grid: robust to bed size, no graph capture) ----
        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size = self.voxel_size
        mpm_options.grid_type = "sparse"
        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50
        mpm_options.critical_fraction = 0.0

        self.mpm_solver = SolverImplicitMPM(self.sand_model, config=mpm_options)
        # read colliders (the ball) from the rigid-body model
        self.mpm_solver.setup_collider(model=self.model)

        # rigid-body solver
        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=False, njmax=100)

        # ---- states ----
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.sand_state_0 = self.sand_model.state()
        self.sand_state_0.body_q = wp.empty_like(self.state_0.body_q)
        self.sand_state_0.body_qd = wp.empty_like(self.state_0.body_qd)
        self.sand_state_0.body_f = wp.empty_like(self.state_0.body_f)

        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # populate body state from joint coords (carries the impact velocity)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ---- two-way coupling buffers ----
        max_nodes = 1 << 20
        self.collider_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_pos = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_ids = wp.full(max_nodes, value=-1, dtype=int, device=self.model.device)
        self.collect_collider_impulses()
        self.collider_body_id = self.mpm_solver.collider_body_index
        self.body_sand_forces = wp.zeros_like(self.state_0.body_f)

        # ---- viewer ----
        self.viewer.set_model(self.model)
        self.particle_render_colors = wp.full(
            self.sand_model.particle_count,
            value=wp.vec3(0.76, 0.70, 0.50),
            dtype=wp.vec3,
            device=self.sand_model.device,
        )
        self.viewer.show_particles = True
        self._frame_camera()

        self.capture()

    def _frame_camera(self):
        """Place the camera close to the impact axis so the drop is visible.

        The scene is only ~decimeters across (surface at ``bed_surface_z``),
        while the viewer's default camera sits meters away. Aim a 3/4 view at a
        point just below the surface, with a distance that scales with the bed.
        """
        if not hasattr(self.viewer, "set_camera"):
            return
        # look a little below the surface, where the ball penetrates and rests
        target_z = self.bed_surface_z - self.ball_diameter
        distance = 2.2 * self.bed_half_width + 3.0 * self.ball_diameter
        eye = (0.7 * distance, -0.7 * distance, target_z + 0.5 * distance)
        to_target = (-eye[0], -eye[1], target_z - eye[2])
        norm = math.sqrt(sum(c * c for c in to_target))
        pitch = math.degrees(math.asin(to_target[2] / norm))
        yaw = math.degrees(math.atan2(to_target[1], to_target[0]))
        self.viewer.set_camera(pos=wp.vec3(*eye), pitch=pitch, yaw=yaw)

    # ------------------------------------------------------------------
    # scene construction
    # ------------------------------------------------------------------
    def _build_ball(self, builder: newton.ModelBuilder) -> int:
        # solid-sphere mass and inertia from the requested density
        radius = self.ball_radius
        mass = self.ball_density * (4.0 / 3.0) * math.pi * radius**3
        inertia_scalar = 0.4 * mass * radius**2
        inertia = wp.mat33(
            inertia_scalar,
            0.0,
            0.0,
            0.0,
            inertia_scalar,
            0.0,
            0.0,
            0.0,
            inertia_scalar,
        )

        # Start the ball a short, visible distance above the surface and launch
        # it downward with a reduced speed so gravity accelerates it to exactly
        # ``impact_speed`` at the surface (v_start^2 = v_impact^2 - 2 g gap).
        # This shows a brief drop without changing the impact conditions.
        v_impact = self.impact_speed
        max_gap = 0.9 * v_impact * v_impact / (2.0 * GRAVITY) if v_impact > 0.0 else 0.0
        gap = max(min(self.approach_gap, max_gap), 1.5 * self.voxel_size)
        v_start = math.sqrt(max(0.0, v_impact * v_impact - 2.0 * GRAVITY * gap))

        start_z = self.bed_surface_z + radius + gap
        qd_start = len(builder.joint_qd)
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, start_z), q=wp.quat_identity()),
            mass=mass,
            inertia=inertia,
        )
        # free-joint DOFs are [vx, vy, vz, wx, wy, wz]; set downward linear velocity
        builder.joint_qd[qd_start + 2] = -v_start

        # density=0 so the shape adds no extra mass on top of the explicit values
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5)
        builder.add_shape_sphere(body, radius=radius, cfg=cfg, color=wp.vec3(0.2, 0.3, 0.8))
        return body

    def _add_container(self, builder: newton.ModelBuilder):
        """Add four static walls that confine the bed (as MPM colliders).

        Without lateral confinement the unconfined sand block slumps into a
        pile; the walls keep a flat, deep bed like the experiment's container.
        """
        hw = self.bed_half_width
        t = max(0.02, self.voxel_size)
        # walls span from slightly below ground (to seal the ground/wall corner)
        # up to a little above the bed surface
        wall_bottom = -3.0 * self.voxel_size
        wall_top = self.bed_depth + 4.0 * self.voxel_size
        wall_half_height = 0.5 * (wall_top - wall_bottom)
        cz = 0.5 * (wall_top + wall_bottom)
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.3)
        for sign in (-1.0, 1.0):
            builder.add_shape_box(
                body=-1,
                cfg=cfg,
                xform=wp.transform(wp.vec3(sign * (hw + t), 0.0, cz), wp.quat_identity()),
                hx=t,
                hy=hw + 2.0 * t,
                hz=wall_half_height,
            )
            builder.add_shape_box(
                body=-1,
                cfg=cfg,
                xform=wp.transform(wp.vec3(0.0, sign * (hw + t), cz), wp.quat_identity()),
                hx=hw + 2.0 * t,
                hy=t,
                hz=wall_half_height,
            )

    def _emit_particles(self, sand_builder: newton.ModelBuilder):
        particles_per_cell = 3.0

        bed_lo = np.array([-self.bed_half_width, -self.bed_half_width, 0.0])
        bed_hi = np.array([self.bed_half_width, self.bed_half_width, self.bed_depth])
        bed_res = np.array(np.ceil(particles_per_cell * (bed_hi - bed_lo) / self.voxel_size), dtype=int)

        cell_size = (bed_hi - bed_lo) / bed_res
        cell_volume = float(np.prod(cell_size))
        radius = float(np.max(cell_size) * 0.5)
        mass = cell_volume * self.sand_density

        sand_builder.add_particle_grid(
            pos=wp.vec3(bed_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(bed_res[0]) + 1,
            dim_y=int(bed_res[1]) + 1,
            dim_z=int(bed_res[2]) + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
            custom_attributes={"mpm:friction": self.sand_friction},
        )

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def capture(self):
        # sparse MPM grid allocates dynamically, so we step eagerly
        self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            wp.launch(
                compute_body_forces,
                dim=self.collider_impulse_ids.shape[0],
                inputs=[
                    self.frame_dt,
                    self.collider_impulse_ids,
                    self.collider_impulses,
                    self.collider_impulse_pos,
                    self.collider_body_id,
                    self.state_0.body_q,
                    self.model.body_com,
                    self.state_0.body_f,
                ],
            )
            self.body_sand_forces.assign(self.state_0.body_f)

            self.viewer.apply_forces(self.state_0)

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.simulate_sand()

    def collect_collider_impulses(self):
        impulses, pos, ids = self.mpm_solver.collect_collider_impulses(self.sand_state_0)
        self.collider_impulse_ids.fill_(-1)
        n = min(impulses.shape[0], self.collider_impulses.shape[0])
        self.collider_impulses[:n].assign(impulses[:n])
        self.collider_impulse_pos[:n].assign(pos[:n])
        self.collider_impulse_ids[:n].assign(ids[:n])

    def simulate_sand(self):
        if self.sand_state_0.body_q is not None:
            wp.launch(
                subtract_body_force,
                dim=self.sand_state_0.body_q.shape,
                inputs=[
                    self.frame_dt,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.body_sand_forces,
                    self.model.body_inv_inertia,
                    self.model.body_inv_mass,
                    self.sand_state_0.body_q,
                    self.sand_state_0.body_qd,
                ],
            )

        self.mpm_solver.step(self.sand_state_0, self.sand_state_0, contacts=None, control=None, dt=self.frame_dt)
        self.collect_collider_impulses()

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        voxel_size = self.mpm_solver.voxel_size

        # essentially all particles stay above ground; tolerate a tiny fraction
        # squeezing below at the container wall corners
        z = self.sand_state_0.particle_q.numpy()[:, 2]
        below = int(np.count_nonzero(z < -3.0 * voxel_size))
        frac = below / max(1, z.shape[0])
        assert frac < 1e-3, f"{below} particles ({frac:.2%}) fell through the ground"

        # the ball must penetrate the bed and come to rest above the ground
        bq = self.state_0.body_q.numpy()[self.ball]
        bqd = self.state_0.body_qd.numpy()[self.ball]
        ball_z = float(bq[2])
        vz = float(bqd[2])
        penetration = self.bed_surface_z - (ball_z - self.ball_radius)
        assert ball_z > -voxel_size, "ball fell through the ground"
        assert penetration > 0.0, f"ball did not penetrate the bed (penetration={penetration:.3f} m)"
        assert abs(vz) < 0.5, f"ball has not come to rest (vz={vz:.2f} m/s)"

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.log_points(
            "/sand",
            points=self.sand_state_0.particle_q,
            radii=self.sand_model.particle_radius,
            colors=self.particle_render_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        # impactor (paper defaults: 1-inch steel sphere, m ~ 69 g)
        parser.add_argument("--ball-density", type=float, default=8070.0, help="Ball density rho_b [kg/m^3].")
        parser.add_argument("--ball-diameter", type=float, default=0.0254, help="Ball diameter D_b [m].")
        parser.add_argument("--drop-height", type=float, default=0.5, help="Drop height H [m] (sets impact speed).")
        parser.add_argument(
            "--approach-gap",
            type=float,
            default=2.0,
            help="Visible drop distance above the surface, in ball diameters (impact speed is preserved).",
        )

        # granular bed (paper defaults: glass beads)
        parser.add_argument("--sand-density", type=float, default=1520.0, help="Sand bulk density [kg/m^3].")
        parser.add_argument("--friction", "-mu", type=float, default=0.45, help="Sand internal friction coefficient.")
        parser.add_argument(
            "--yield-pressure",
            "-yp",
            type=float,
            default=1.0e6,
            help="Drucker-Prager yield pressure [Pa]. High enough that friction grows with depth.",
        )
        parser.add_argument(
            "--young-modulus", "-ym", type=float, default=1.0e7, help="Sand elastic (Young) modulus [Pa]."
        )
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.005, help="MPM grid voxel size [m].")
        parser.add_argument("--bed-width-diameters", type=float, default=4.0, help="Bed half-width in ball diameters.")
        parser.add_argument("--bed-depth-diameters", type=float, default=6.0, help="Bed depth in ball diameters.")
        parser.add_argument("--fps", type=float, default=500.0, help="Frames per second (fine sampling of the impact).")
        parser.add_argument("--substeps", type=int, default=2, help="Rigid substeps per frame.")

        # force-law analysis (Katsuragi & Durian 2007)
        parser.add_argument(
            "--force-law",
            action="store_true",
            help="Record projectile dynamics over several drop heights and fit F = -mg + k|z| + m v^2/d1.",
        )
        parser.add_argument(
            "--force-law-heights",
            type=float,
            nargs="+",
            default=[0.1, 0.2, 0.35, 0.5, 0.7, 0.9],
            help="Drop heights [m] to sample for the force-law fit.",
        )
        parser.add_argument(
            "--force-law-depths",
            type=float,
            nargs="+",
            default=[0.8, 1.2, 1.6, 2.0, 2.4],
            help="Fixed penetration depths (in ball diameters) at which to sample (v, F) for the fit.",
        )
        parser.add_argument(
            "--force-law-plot", type=str, default=None, help="Optional path to save the force-law diagnostic plots."
        )

        # stopping-time experiment (paper's hallmark: t_stop decreases with v0)
        parser.add_argument(
            "--stopping-time",
            action="store_true",
            help="Sweep impact speeds and plot stopping time vs impact speed.",
        )
        parser.add_argument(
            "--stopping-time-speeds",
            type=float,
            nargs="+",
            default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
            help="Impact speeds v0 [m/s] to sample for the stopping-time experiment.",
        )
        parser.add_argument(
            "--stopping-time-plot",
            type=str,
            default="mpm_impact_stopping_time.png",
            help="Path to save the stopping-time-vs-speed plot.",
        )
        parser.add_argument(
            "--paper-d1",
            type=float,
            default=0.087,
            help="Paper's inertial-drag length d1 [m] for the analytical stopping-time curve.",
        )
        parser.add_argument(
            "--paper-k-over-m",
            type=float,
            default=1040.0,
            help="Paper's friction coefficient k/m [s^-2] for the analytical stopping-time curve.",
        )
        return parser


# ----------------------------------------------------------------------
# Force-law analysis (Katsuragi & Durian, Nature Physics 2007)
# ----------------------------------------------------------------------
def _record_impact(example: Example, frames: int) -> dict:
    """Step the impact and record the ball's vertical dynamics each frame.

    Returns dict of arrays: ``t`` [s], ``z`` (ball centre height [m]),
    ``vz`` (vertical velocity [m/s], negative downward) and ``fz`` (vertical
    sand force on the ball [N], equal to ``m*(a + g)``).
    """
    t, z, vz, fz = [], [], [], []
    for i in range(frames):
        example.step()
        bq = example.state_0.body_q.numpy()[example.ball]
        bqd = example.state_0.body_qd.numpy()[example.ball]
        fsand = example.body_sand_forces.numpy()[example.ball]
        t.append((i + 1) * example.frame_dt)
        z.append(float(bq[2]))
        vz.append(float(bqd[2]))
        fz.append(float(fsand[2]))
    return {"t": np.array(t), "z": np.array(z), "vz": np.array(vz), "fz": np.array(fz)}


def run_force_law(args):
    import copy  # noqa: PLC0415

    import newton.viewer  # noqa: PLC0415

    depths_db = np.asarray(args.force_law_depths, dtype=float)

    print("\n=== unified force law  F = -mg + k|z| + m v^2/d1 ===")
    print(f"drop heights [m]: {args.force_law_heights}")
    print(f"sampling depths [D_b]: {list(depths_db)}\n")
    print(f"{'v0 [m/s]':>9} {'t_stop [s]':>11} {'pen [cm]':>9} {'pen/D_b':>8}")

    di, vv, ff = [], [], []  # depth index, downward speed, sand force
    stop_rows = []
    mass = ball_diameter = sand_density = 0.0
    for height in args.force_law_heights:
        run_args = copy.deepcopy(args)
        run_args.drop_height = float(height)

        viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)
        example = Example(viewer, run_args)
        mass = float(example.model.body_mass.numpy()[example.ball])
        ball_diameter = example.ball_diameter
        sand_density = example.sand_density
        radius = example.ball_radius
        surface = example.bed_surface_z

        rec = _record_impact(example, args.num_frames)
        viewer.close()

        pen = surface - (rec["z"] - radius)
        vz = rec["vz"]
        mask = (pen > 0.003) & (vz < -0.05)
        v0 = impact_speed(float(height))
        if mask.sum() >= 5:
            depths = depths_db * ball_diameter
            p, vd, force = pen[mask], -vz[mask], _smooth(rec["fz"][mask])
            order = np.argsort(p)
            p, vd, force = p[order], vd[order], force[order]
            for i, z_i in enumerate(depths):
                if p.min() <= z_i <= p.max():
                    di.append(i)
                    vv.append(float(np.interp(z_i, p, vd)))
                    ff.append(float(np.interp(z_i, p, force)))
            # stopping time = duration from surface entry to rest (robust to the
            # visible approach gap, which shifts absolute frame times)
            t_stop = float(rec["t"][mask][-1] - rec["t"][mask][0])
            stop_rows.append((v0, t_stop, pen.max()))
            print(f"{v0:>9.2f} {t_stop:>11.4f} {pen.max() * 100:>9.2f} {pen.max() / ball_diameter:>8.1f}")
        else:
            print(f"{v0:>9.2f} {'-':>11} {'-':>9} {'-':>8}  (insufficient penetration samples)")

    di = np.asarray(di)
    vv = np.asarray(vv)
    ff = np.asarray(ff)
    depths = depths_db * ball_diameter
    n = len(depths)
    if ff.size < n + 2:
        print("\nNot enough samples to fit the force law. Try more/deeper drop heights or finer voxels.")
        return

    # Fig. 3a fit: F = F(z_i) + (m/d1) v^2 with a shared d1 and per-depth intercepts.
    design = np.zeros((ff.size, n + 1))
    design[np.arange(ff.size), di] = 1.0
    design[:, n] = vv**2
    coef, *_ = np.linalg.lstsq(design, ff, rcond=None)
    friction = coef[:n]  # F(z_i)  [N]
    beta = coef[n]  # m / d1
    pred = design @ coef
    ss_res = float(np.sum((ff - pred) ** 2))
    ss_tot = float(np.sum((ff - ff.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    d1 = mass / beta if beta > 0 else float("nan")
    drag_c = beta / (sand_density * ball_diameter**2)  # F_drag = C rho_g D_b^2 v^2
    d1_pred = mass / (0.8 * sand_density * ball_diameter**2)

    # Fig. 3b: friction F(z_i) should grow linearly with depth -> slope k
    valid = np.isfinite(friction)
    k_slope = np.polyfit(depths[valid], friction[valid], 1)[0] if valid.sum() >= 2 else float("nan")

    print("\n--- force-law fit (Katsuragi & Durian Fig. 3 method) ---")
    print(f"points = {ff.size}   R^2 = {r_squared:.3f}")
    print(f"inertial drag: d1 = {d1 * 100:.2f} cm (constant across depths)")
    print(f"               predicted m/(0.8 rho_g D_b^2) = {d1_pred * 100:.2f} cm")
    print(f"               drag coefficient C = {drag_c:.3f}  (paper interpretation ~0.8)")
    print(f"friction:      k/m = {k_slope / mass:.1f} s^-2  (linear-in-depth Coulomb term)")
    print("               F(z_i)/m by depth:")
    for z_i, f_i in zip(depths, friction, strict=False):
        print(f"                 z = {z_i * 100:5.2f} cm   F/m = {f_i / mass:8.1f} s^-2")
    if stop_rows and stop_rows[0][1] > stop_rows[-1][1]:
        print("stopping time DECREASES with impact speed (matches the paper's hallmark result).")

    if args.force_law_plot:
        _plot_force_law(args.force_law_plot, di, vv, ff, depths, friction, beta, mass, stop_rows)


def _plot_force_law(path, di, vv, ff, depths, friction, beta, mass, stop_rows):
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) F vs v^2 at fixed depths + shared-slope lines
    for i, z_i in enumerate(depths):
        sel = di == i
        if np.any(sel):
            color = f"C{i}"
            ax0.plot(vv[sel] ** 2, ff[sel], "o", color=color, label=f"z={z_i * 100:.1f} cm")
            vs = np.linspace(0, (vv[sel] ** 2).max(), 20)
            ax0.plot(vs, friction[i] + beta * vs, "-", color=color, lw=1)
    ax0.set_xlabel(r"$v^2$ [m$^2$/s$^2$]")
    ax0.set_ylabel(r"sand force $F$ [N]")
    ax0.set_title("(a) inertial drag: slope $m/d_1$ shared")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)

    # (b) friction law F(z_i) vs depth
    ax1.plot(depths * 100, friction / mass, "o-")
    ax1.set_xlabel("depth $z$ [cm]")
    ax1.set_ylabel(r"$F(z)/m$ [s$^{-2}$]")
    ax1.set_title("(b) depth-dependent friction $k|z|$")
    ax1.grid(alpha=0.3)

    # (c) stopping time vs impact speed
    if stop_rows:
        v0s = [r[0] for r in stop_rows]
        ts = [r[1] for r in stop_rows]
        ax2.plot(v0s, ts, "s-")
    ax2.set_xlabel(r"impact speed $v_0$ [m/s]")
    ax2.set_ylabel(r"stopping time $t_{stop}$ [s]")
    ax2.set_title("(c) stopping time vs impact speed")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"saved force-law plots to {path}")


# ----------------------------------------------------------------------
# Stopping-time experiment: t_stop vs impact speed (the paper's hallmark)
# ----------------------------------------------------------------------
def _stopping_time_and_depth(rec: dict, surface: float, radius: float) -> tuple[float | None, float]:
    """Return (stopping time [s], max penetration [m]) from a recorded impact.

    Stopping time is the duration from surface entry to the *first* time the
    ball ceases moving downward (robust to the approach gap and to any late slow
    settling/creep after the ball has essentially stopped). Returns
    ``(None, pen_max)`` if the ball did not clearly penetrate and stop.
    """
    pen = surface - (rec["z"] - radius)
    vz = rec["vz"]
    pen_max = float(pen.max())

    entered = np.where((pen > 0.003) & (vz < -0.05))[0]
    if entered.size < 1:
        return None, pen_max
    i0 = int(entered[0])
    # first frame at/after entry where downward motion has ceased
    stopped = np.where(vz[i0:] >= -0.05)[0]
    if stopped.size < 1:
        return None, pen_max  # never came to rest inside the recording window
    i1 = i0 + int(stopped[0])
    return float(rec["t"][i1] - rec["t"][i0]), pen_max


def _analytical_stopping_time(v0: float, d1: float, k_over_m: float, g: float = GRAVITY) -> float:
    """Stopping time from the paper's force law, integrated with RK4.

    Solves ``dv/dt = g - v^2/d1 - (k/m) z`` (mass cancels) from surface entry
    (``z=0``, ``v=v0``) until the ball's downward speed returns to zero, and
    returns that time [s]. See Katsuragi & Durian (2007), Eq. (1).
    """
    dt = 1.0e-5
    t_max = 2.0

    def acc(z: float, v: float) -> float:
        return g - v * v / d1 - k_over_m * z

    z, v, t = 0.0, v0, 0.0
    while v > 0.0 and t < t_max:
        k1z, k1v = v, acc(z, v)
        k2z, k2v = v + 0.5 * dt * k1v, acc(z + 0.5 * dt * k1z, v + 0.5 * dt * k1v)
        k3z, k3v = v + 0.5 * dt * k2v, acc(z + 0.5 * dt * k2z, v + 0.5 * dt * k2v)
        k4z, k4v = v + dt * k3v, acc(z + dt * k3z, v + dt * k3v)
        v_new = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        if v_new <= 0.0:
            return t + dt * v / (v - v_new)  # linear interpolation of the v=0 crossing
        z += (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
        v = v_new
        t += dt
    return t


def run_stopping_time(args):
    import copy  # noqa: PLC0415

    import newton.viewer  # noqa: PLC0415

    speeds = sorted(float(s) for s in args.stopping_time_speeds)
    d1, kom = float(args.paper_d1), float(args.paper_k_over_m)

    print("\n=== stopping time vs impact speed ===")
    print(f"paper model: d1 = {d1 * 100:.1f} cm, k/m = {kom:.0f} s^-2")
    print(f"{'v0 [m/s]':>9} {'H [cm]':>8} {'t_sim [ms]':>11} {'t_paper [ms]':>13} {'pen [cm]':>9}")

    v0s, tstops = [], []
    for v0 in speeds:
        run_args = copy.deepcopy(args)
        run_args.drop_height = v0 * v0 / (2.0 * GRAVITY)
        # no visible drop needed here; impact speed at the surface is preserved
        run_args.approach_gap = 0.0

        viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)
        example = Example(viewer, run_args)
        radius = example.ball_radius
        surface = example.bed_surface_z
        rec = _record_impact(example, args.num_frames)
        viewer.close()

        t_pred = _analytical_stopping_time(v0, d1, kom)
        t_stop, pen_max = _stopping_time_and_depth(rec, surface, radius)
        if t_stop is None:
            print(
                f"{v0:>9.2f} {run_args.drop_height * 100:>8.2f} {'insufficient':>11} "
                f"{t_pred * 1000:>13.1f} {pen_max * 100:>9.2f}"
            )
            continue
        v0s.append(v0)
        tstops.append(t_stop)
        print(
            f"{v0:>9.2f} {run_args.drop_height * 100:>8.2f} {t_stop * 1000:>11.1f} "
            f"{t_pred * 1000:>13.1f} {pen_max * 100:>9.2f}"
        )

    if len(v0s) < 2:
        print("\nNot enough valid impacts to plot. Try larger speeds or more frames.")
        return

    trend = "DECREASES" if tstops[-1] < tstops[0] else "increases"
    print(f"\nstopping time {trend} as impact speed increases (paper: decreases).")
    _plot_stopping_time(args.stopping_time_plot, v0s, tstops, d1, kom)


def _plot_stopping_time(path: str, v0s: list[float], tstops: list[float], d1: float, k_over_m: float):
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    # analytical curve from the paper's force law over the sampled speed range
    v_fine = np.linspace(min(v0s), max(v0s), 100)
    t_fine = np.array([_analytical_stopping_time(float(v), d1, k_over_m) for v in v_fine])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(v0s, np.asarray(tstops) * 1000.0, "o-", color="C3", lw=1.5, ms=7, label="Newton MPM")
    ax.plot(
        v_fine,
        t_fine * 1000.0,
        "--",
        color="C0",
        lw=1.8,
        label=f"paper force law ($d_1$={d1 * 100:.1f} cm, $k/m$={k_over_m:.0f} s$^{{-2}}$)",
    )
    ax.set_xlabel(r"impact speed $v_0$ [m/s]")
    ax.set_ylabel(r"stopping time $t_\mathrm{stop}$ [ms]")
    ax.set_title("Granular impact: stopping time vs impact speed")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"saved stopping-time plot to {path}")


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    if args.force_law or args.stopping_time:
        # these analyses manage their own headless (null) viewers per run
        import newton.viewer

        viewer.close()
        if args.force_law:
            run_force_law(args)
        else:
            run_stopping_time(args)
    else:
        newton.examples.run(Example(viewer, args), args)
