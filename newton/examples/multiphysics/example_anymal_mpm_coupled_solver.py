# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ANYmal-MPM Coupled Solver
#
# Shows ANYmal C with a pretrained policy and bidirectional implicit-MPM
# coupling through lagged proxy-body feedback.
#
# Command: python -m newton.examples anymal_mpm_coupled_solver
#
###########################################################################

import sys

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy
from warp_nn.runtime import OnnxRuntime

import newton
import newton.examples
import newton.utils
from newton.examples.robot.example_robot_anymal_c_walk import (
    _build_joint_target_q_kernel,
    _compute_obs_kernel,
    lab_to_mujoco,
    mujoco_to_lab,
)
from newton.examples.robot.onnx_policy_utils import validate_policy_io_shapes
from newton.solvers import SolverImplicitMPM


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        voxel_size = args.voxel_size
        particles_per_cell = args.particles_per_cell
        tolerance = args.tolerance
        grid_type = args.grid_type
        gravel_depth = args.gravel_depth
        gravel_length = args.gravel_length
        foot_embed_depth = args.foot_embed_depth
        if args.physics_substeps < 1:
            raise ValueError("physics_substeps must be at least 1")
        if gravel_depth <= 0.0:
            raise ValueError("gravel_depth must be positive")
        if gravel_length <= 0.0:
            raise ValueError("gravel_length must be positive")
        if foot_embed_depth < 0.0 or foot_embed_depth > gravel_depth:
            raise ValueError("foot_embed_depth must be nonnegative and no greater than gravel_depth")

        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.physics_substeps = args.physics_substeps
        self.sim_dt = self.frame_dt / self.physics_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_c")
        stage_path = str(asset_path / "urdf" / "anymal.urdf")
        robot_body_start = builder.body_count
        robot_base_height = 0.62 + gravel_depth - foot_embed_depth + args.spawn_clearance
        builder.add_urdf(
            stage_path,
            xform=wp.transform(
                wp.vec3(0.0, 0.0, robot_base_height),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
            ),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )
        robot_body_end = builder.body_count

        # Only the shanks interact with MPM particles.
        for body in range(robot_body_start, robot_body_end):
            if "SHANK" not in builder.body_label[body]:
                for shape in builder.body_shapes[body]:
                    builder.shape_flags[shape] = builder.shape_flags[shape] & ~newton.ShapeFlags.COLLIDE_PARTICLES

        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=args.mpm_friction))

        initial_q = {
            "RH_HAA": 0.0,
            "RH_HFE": -0.4,
            "RH_KFE": 0.8,
            "LH_HAA": 0.0,
            "LH_HFE": -0.4,
            "LH_KFE": 0.8,
            "RF_HAA": 0.0,
            "RF_HFE": 0.4,
            "RF_KFE": -0.8,
            "LF_HAA": 0.0,
            "LF_HFE": 0.4,
            "LF_KFE": -0.8,
        }
        for name, value in initial_q.items():
            idx = next(i for i, lbl in enumerate(builder.joint_label) if lbl.endswith(f"/{name}"))
            builder.joint_q[idx + 6] = value

        for i in range(builder.joint_dof_count):
            builder.joint_target_ke[i] = 150
            builder.joint_target_kd[i] = 5

        SolverImplicitMPM.register_custom_attributes(builder)

        mpm_particle_start = builder.particle_count
        density = 2500.0
        particle_lo = np.array([-0.5, -0.5, 0.0])
        particle_hi = np.array([0.5, gravel_length - 0.5, gravel_depth])
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )
        _spawn_particles(
            builder,
            particle_res,
            particle_lo,
            particle_hi,
            density,
            custom_attributes={
                "mpm:viscosity": args.mpm_viscosity,
                "mpm:yield_stress": args.mpm_yield_stress,
                "mpm:yield_pressure": args.mpm_yield_pressure,
                "mpm:damping": args.mpm_damping,
                "mpm:friction": args.mpm_friction,
            },
        )

        self.model = builder.finalize()

        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size = voxel_size
        mpm_options.tolerance = tolerance
        mpm_options.transfer_scheme = "pic"
        mpm_options.grid_type = grid_type
        mpm_options.grid_padding = 50 if grid_type == "fixed" else 0
        mpm_options.max_active_cell_count = 1 << 16 if grid_type == "fixed" else -1
        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50
        mpm_options.critical_fraction = 0.0
        mpm_options.air_drag = 1.0
        mpm_options.collider_velocity_mode = "forward"

        robot_body_ids = list(range(robot_body_start, robot_body_end))
        mpm_particle_ids = list(range(mpm_particle_start, self.model.particle_count))

        self.state_0 = self.model.state()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        assert self.state_0.body_q is not None
        self.initial_base_position = self.state_0.body_q.numpy()[0, :3].copy()

        settling_steps = int(np.ceil(max(0.0, args.settling_time) / self.sim_dt))
        if settling_steps > 0:
            settling_solver = SolverImplicitMPM(model=self.model, config=mpm_options)
            settling_contacts = self.model.contacts()
            settling_control = self.model.control()
            settling_solver.setup_collider(
                body_mass=wp.zeros_like(self.model.body_mass),
                body_q=self.state_0.body_q,
            )
            for _ in range(settling_steps):
                settling_solver.step(
                    self.state_0,
                    self.state_0,
                    contacts=settling_contacts,
                    control=settling_control,
                    dt=self.sim_dt,
                )
            del settling_solver

        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="mjc",
                    solver=lambda view: newton.solvers.SolverMuJoCo(
                        model=view,
                        solver="newton",
                        ls_iterations=50,
                        njmax=50,
                        nconmax=100,
                        use_mujoco_contacts=False,
                    ),
                    bodies=robot_body_ids,
                    joints=list(range(self.model.joint_count)),
                    substeps=args.rigid_substeps,
                ),
                SolverCoupledProxy.Entry(
                    name="mpm",
                    solver=lambda view: SolverImplicitMPM(model=view, config=mpm_options),
                    particles=mpm_particle_ids,
                    in_place=True,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mjc",
                        destination="mpm",
                        bodies=robot_body_ids,
                        mass_scale=args.mass_scale,
                        mode="lagged",
                        proxy_relaxation=args.proxy_relaxation,
                        # MPM resolves collider contact internally.
                        collision_pipeline=lambda _model: None,
                    )
                ],
                iterations=args.proxy_iterations,
            ),
        )
        self.mpm_solver = self.solver.solver("mpm")

        self.rigid_collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_max=0)
        self.contacts = self.rigid_collision_pipeline.contacts()

        # Refresh collider arrays after the coupler installs articulated effective inertia.
        self.mpm_solver.setup_collider(model=self.solver.view("mpm"))

        self.control = self.model.control()

        policy_path = str(asset_path / "rl_policies" / "anymal_walking_policy_physx.onnx")
        self.policy = OnnxRuntime(policy_path, device=self.device)
        self._policy_input_name = self.policy.input_names[0]
        self._policy_output_name = self.policy.output_names[0]
        validate_policy_io_shapes(
            policy_path,
            self._policy_input_name,
            self._policy_output_name,
            obs_width=48,
            action_width=12,
            context="example_anymal_mpm_coupled_solver",
        )

        assert self.state_0.joint_q is not None
        self._joint_pos_initial_wp = wp.clone(self.state_0.joint_q[7:])
        self._lab_to_mujoco_wp = wp.array(np.asarray(lab_to_mujoco, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._mujoco_to_lab_wp = wp.array(np.asarray(mujoco_to_lab, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._gravity_w = wp.vec3(0.0, 0.0, -1.0)
        self._command = wp.vec3(0.0, 0.0, 0.0)
        self._obs_wp = wp.zeros((1, 48), dtype=wp.float32, device=self.device)
        self._prev_act_wp = wp.zeros((1, 12), dtype=wp.float32, device=self.device)
        self._auto_forward = True
        self._forward_speed = args.forward_speed

        newton.examples.configure_coupled_view(self, args)
        self.viewer.show_particles = True
        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda and self.mpm_solver.grid_type == "fixed":
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def apply_control(self):
        wp.launch(
            _compute_obs_kernel,
            dim=1,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self._joint_pos_initial_wp,
                self._lab_to_mujoco_wp,
                self._gravity_w,
                self._command,
                self._prev_act_wp,
                self._obs_wp,
            ],
            device=self.device,
        )
        out = self.policy({self._policy_input_name: self._obs_wp})
        act_wp = out[self._policy_output_name]

        wp.launch(
            _build_joint_target_q_kernel,
            dim=19,
            inputs=[
                act_wp,
                self._joint_pos_initial_wp,
                self._mujoco_to_lab_wp,
                0.5,
                7,
                self.control.joint_target_q,
            ],
            device=self.device,
        )
        wp.copy(self._prev_act_wp, act_wp)

    def simulate(self):
        for _ in range(self.physics_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.rigid_collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.sim_dt)

    def step(self):
        if hasattr(self.viewer, "is_key_down"):
            fwd = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lat = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rot = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)

            if fwd or lat or rot:
                self._auto_forward = False

            self._command = wp.vec3(float(fwd), float(lat), float(rot))

        if self._auto_forward:
            self._command = wp.vec3(self._forward_speed, 0.0, 0.0)

        self.apply_control()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot base remains above the ground",
            lambda q, qd: q[2] > 0.1,
            indices=[0],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot base remains upright",
            lambda q, qd: wp.quat_rotate(wp.transform_get_rotation(q), wp.vec3(0.0, 0.0, 1.0))[2] > 0.8,
            indices=[0],
        )
        velocity_min = wp.spatial_vector(-5.0, -5.0, -5.0, -15.0, -15.0, -15.0)
        velocity_max = wp.spatial_vector(5.0, 5.0, 5.0, 15.0, 15.0, 15.0)
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot base velocity remains bounded",
            lambda q, qd: newton.math.vec_inside_limits(qd, velocity_min, velocity_max),
            indices=[0],
        )

        voxel_size = self.mpm_solver.voxel_size
        assert self.state_0.particle_q is not None
        particle_q = self.state_0.particle_q.numpy()
        assert np.isfinite(particle_q).all(), "Particle positions contain NaN or inf values"
        penetration_fraction = np.mean(particle_q[:, 2] < -2.0 * voxel_size)
        assert penetration_fraction < 1.0e-3, "Too many gravel particles penetrated the ground"
        assert self.state_0.body_q is not None
        base_position = self.state_0.body_q.numpy()[0, :3]
        assert base_position[2] > 0.3, "The robot base sank into the gravel"
        assert base_position[1] > self.initial_base_position[1] + 0.5, "The robot did not move forward"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.03)
        parser.add_argument("--particles-per-cell", "-ppc", type=float, default=3.0)
        parser.add_argument("--grid-type", "-gt", choices=["sparse", "dense", "fixed"], default="sparse")
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)
        parser.add_argument(
            "--physics-substeps",
            help="Number of coupled MuJoCo-MPM steps per 50 Hz policy update",
            type=int,
            default=4,
        )
        parser.add_argument(
            "--settling-time",
            help="Duration to pre-settle gravel against a kinematic robot [s]",
            type=float,
            default=0.5,
        )
        parser.add_argument(
            "--gravel-depth",
            help="Depth of deformable gravel above the ground [m]",
            type=float,
            default=0.1,
        )
        parser.add_argument(
            "--foot-embed-depth",
            help="Initial foot penetration into the gravel surface [m]",
            type=float,
            default=0.03,
        )
        parser.add_argument(
            "--gravel-length",
            help="Length of the deformable gravel layer [m]",
            type=float,
            default=5.0,
        )
        parser.add_argument(
            "--spawn-clearance",
            help="Additional robot clearance above the gravel surface [m]",
            type=float,
            default=0.0,
        )
        parser.add_argument(
            "--forward-speed",
            help="Automatic forward velocity command [m/s]",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--mpm-viscosity",
            help="Plastic viscosity of the MPM grains [Pa·s]",
            type=float,
            default=1.0e4,
        )
        parser.add_argument(
            "--mpm-yield-stress",
            help="Cohesive deviatoric yield stress of the MPM grains [Pa]",
            type=float,
            default=1.0e6,
        )
        parser.add_argument(
            "--mpm-yield-pressure",
            help="Compressive yield pressure of the MPM grains [Pa]",
            type=float,
            default=1.0e7,
        )
        parser.add_argument(
            "--mpm-damping",
            help="Elastic damping relaxation time of the MPM grains [s]",
            type=float,
            default=0.02,
        )
        parser.add_argument(
            "--mpm-friction",
            help="Internal friction coefficient of the MPM grains",
            type=float,
            default=0.8,
        )
        parser.add_argument(
            "--rigid-substeps",
            help="Number of MuJoCo substeps per coupled physics step",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--proxy-iterations",
            help="Number of proxy relaxation passes per coupled step",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--mass-scale",
            help="Scale factor for articulated effective mass used by MPM proxies",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--proxy-relaxation",
            help="Relaxation factor for MPM feedback wrenches",
            type=float,
            default=1.0,
        )
        return parser


def _spawn_particles(builder: newton.ModelBuilder, res, bounds_lo, bounds_hi, density, custom_attributes):
    cell_size = (bounds_hi - bounds_lo) / res
    cell_volume = np.prod(cell_size)
    radius = np.max(cell_size) * 0.5
    mass = np.prod(cell_volume) * density

    builder.add_particle_grid(
        pos=wp.vec3(bounds_lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=res[0] + 1,
        dim_y=res[1] + 1,
        dim_z=res[2] + 1,
        cell_x=cell_size[0],
        cell_y=cell_size[1],
        cell_z=cell_size[2],
        mass=mass,
        jitter=0.25 * radius,
        radius_mean=radius,
        custom_attributes=custom_attributes,
    )


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    if wp.get_device().is_cpu:
        print("Error: This example requires a GPU device.")
        sys.exit(1)

    newton.examples.run(Example(viewer, args), args)
