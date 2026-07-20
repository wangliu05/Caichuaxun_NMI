import numpy as np
from scipy.optimize import OptimizeResult, minimize


class InverseActuationSolver:
    @classmethod
    def from_config(cls, field_model, config):
        solver_name = str(getattr(config, "INVERSE_SOLVER", "auto")).lower()
        if solver_name not in ("auto", "casadi", "scipy"):
            raise ValueError("INVERSE_SOLVER must be 'auto', 'casadi', or 'scipy'")

        if solver_name in ("auto", "casadi"):
            try:
                import casadi as ca
                return CasadiInverseActuationSolver.from_config(field_model, config, ca)
            except ImportError:
                if solver_name == "casadi":
                    raise RuntimeError("INVERSE_SOLVER='casadi' requires: pip install casadi")
                print("[InverseSolver] casadi is not installed; using SciPy SLSQP fallback")

        return ScipyInverseActuationSolver.from_config(field_model, config)


class ScipyInverseActuationSolver:
    def __init__(self, field_model, maxiter=80, lambda_dq=1e-4, max_step_rad=np.pi / 6, min_particle_field_t=0.015):
        self.model = field_model
        self.maxiter = maxiter
        self.lambda_dq = lambda_dq
        self.max_step_rad = max_step_rad
        self.min_particle_field_t = min_particle_field_t
        self.last_q = np.zeros(self.model.array.magnet_count * 2, dtype=float)
        self.solver_name = "scipy-slsqp"

    @classmethod
    def from_config(cls, field_model, config):
        return cls(
            field_model,
            maxiter=config.INVERSE_MAXITER,
            lambda_dq=config.INVERSE_LAMBDA_DQ,
            max_step_rad=config.INVERSE_MAX_STEP_RAD,
            min_particle_field_t=config.MIN_PARTICLE_FIELD_T,
        )

    def solve(self, objective, q0=None):
        q_prev = self.last_q if q0 is None else np.asarray(q0, dtype=float).reshape(-1)
        if objective.kind == "guidewire_field":
            result = self._solve_guidewire(objective.position_m, objective.desired_field_t, q_prev)
        elif objective.kind == "particle_gradient_drive":
            result = self._solve_particle(objective, q_prev)
        elif objective.kind == "field_cancellation":
            result = self._solve_field_cancellation(objective.metadata["points_m"], q_prev)
        else:
            raise ValueError(f"Unknown objective kind: {objective.kind}")

        if result.success:
            self.last_q = np.asarray(result.x, dtype=float)
        return result

    def _bounds(self, q_prev):
        lower = np.maximum(-np.pi, q_prev - self.max_step_rad)
        upper = np.minimum(np.pi, q_prev + self.max_step_rad)
        return list(zip(lower, upper))

    def _solve_guidewire(self, p_m, b_desired_t, q_prev):
        p = np.asarray(p_m, dtype=float)
        b_desired = np.asarray(b_desired_t, dtype=float)

        def cost(q):
            b = self.model.field(p, q)
            return float(np.sum((b - b_desired) ** 2) + self.lambda_dq * np.sum((q - q_prev) ** 2))

        return minimize(
            cost,
            q_prev,
            method="SLSQP",
            bounds=self._bounds(q_prev),
            options={"maxiter": self.maxiter, "ftol": 1e-12},
        )

    def _solve_particle(self, objective, q_prev):
        p = np.asarray(objective.position_m, dtype=float)
        desired_drive = np.asarray(objective.desired_gradient_drive_n, dtype=float)

        def cost(q):
            b = self.model.field(p, q)
            b_norm = np.linalg.norm(b)
            b_hat = b / max(b_norm, 1e-12)
            drive = self.model.gradient(p, q).T @ b_hat
            field_penalty = max(0.0, self.min_particle_field_t - b_norm) ** 2
            return float(
                np.sum((drive - desired_drive) ** 2)
                + 10.0 * field_penalty
                + self.lambda_dq * np.sum((q - q_prev) ** 2)
            )

        return minimize(
            cost,
            q_prev,
            method="SLSQP",
            bounds=self._bounds(q_prev),
            options={"maxiter": self.maxiter, "ftol": 1e-12},
        )

    def _solve_field_cancellation(self, points_m, q_prev):
        pts = np.asarray(points_m, dtype=float)
        if np.allclose(q_prev, 0.0):
            rng = np.random.default_rng(7)
            q_start = rng.uniform(-np.pi, np.pi, size=q_prev.shape)
        else:
            q_start = q_prev

        def cost(q):
            fields = self.model.fields(pts, q)
            return float(np.mean(np.sum(fields * fields, axis=1)))

        bounds = [(-np.pi, np.pi) for _ in range(len(q_prev))]
        return minimize(cost, q_start, method="SLSQP", bounds=bounds, options={"maxiter": self.maxiter, "ftol": 1e-12})


class CasadiInverseActuationSolver:
    def __init__(self, field_model, ca, maxiter=80, lambda_dq=1e-4, max_step_rad=np.pi / 6,
                 min_particle_field_t=0.015, print_level=0):
        self.model = field_model
        self.array = field_model.array
        self.ca = ca
        self.maxiter = int(maxiter)
        self.lambda_dq = float(lambda_dq)
        self.max_step_rad = float(max_step_rad)
        self.min_particle_field_t = float(min_particle_field_t)
        self.print_level = int(print_level)
        self.last_q = np.zeros(self.array.magnet_count * 2, dtype=float)
        self.solver_name = "casadi-ipopt"

    @classmethod
    def from_config(cls, field_model, config, ca):
        return cls(
            field_model,
            ca,
            maxiter=config.INVERSE_MAXITER,
            lambda_dq=config.INVERSE_LAMBDA_DQ,
            max_step_rad=config.INVERSE_MAX_STEP_RAD,
            min_particle_field_t=config.MIN_PARTICLE_FIELD_T,
            print_level=getattr(config, "IPOPT_PRINT_LEVEL", 0),
        )

    def solve(self, objective, q0=None):
        q_prev = self.last_q if q0 is None else np.asarray(q0, dtype=float).reshape(-1)
        if objective.kind == "guidewire_field":
            result = self._solve_guidewire(objective.position_m, objective.desired_field_t, q_prev)
        elif objective.kind == "particle_gradient_drive":
            result = self._solve_particle(objective, q_prev)
        elif objective.kind == "field_cancellation":
            result = self._solve_field_cancellation(objective.metadata["points_m"], q_prev)
        else:
            raise ValueError(f"Unknown objective kind: {objective.kind}")

        if result.success:
            self.last_q = np.asarray(result.x, dtype=float)
        return result

    def _solve_guidewire(self, p_m, b_desired_t, q_prev):
        ca = self.ca
        q = ca.SX.sym("q", len(q_prev))
        b = self._field_expr(np.asarray(p_m, dtype=float), q)
        b_desired = ca.DM(np.asarray(b_desired_t, dtype=float).reshape(3))
        dq = q - ca.DM(q_prev)
        cost = ca.sumsqr(b - b_desired) + self.lambda_dq * ca.sumsqr(dq)
        return self._solve_nlp(q, cost, q_prev)

    def _solve_particle(self, objective, q_prev):
        ca = self.ca
        q = ca.SX.sym("q", len(q_prev))
        p = np.asarray(objective.position_m, dtype=float)
        b = self._field_expr(p, q)
        b_norm = ca.sqrt(ca.dot(b, b) + 1e-24)
        b_hat = b / b_norm
        drive = ca.mtimes(self._gradient_expr(p, q).T, b_hat)
        desired_drive = ca.DM(np.asarray(objective.desired_gradient_drive_n, dtype=float).reshape(3))
        dq = q - ca.DM(q_prev)
        cost = ca.sumsqr(drive - desired_drive) + self.lambda_dq * ca.sumsqr(dq)
        return self._solve_nlp(q, cost, q_prev, g=b_norm, lbg=self.min_particle_field_t, ubg=ca.inf)

    def _solve_field_cancellation(self, points_m, q_prev):
        ca = self.ca
        if np.allclose(q_prev, 0.0):
            rng = np.random.default_rng(7)
            q_start = rng.uniform(-np.pi, np.pi, size=q_prev.shape)
            full_bounds = True
        else:
            q_start = q_prev
            full_bounds = False

        q = ca.SX.sym("q", len(q_prev))
        costs = []
        for p in np.asarray(points_m, dtype=float):
            costs.append(ca.sumsqr(self._field_expr(p, q)))
        cost = sum(costs) / max(len(costs), 1)
        return self._solve_nlp(q, cost, q_start, full_bounds=full_bounds)

    def _solve_nlp(self, q, cost, q_start, g=None, lbg=None, ubg=None, full_bounds=False):
        ca = self.ca
        nlp = {"x": q, "f": cost}
        if g is not None:
            nlp["g"] = g
        opts = {
            "ipopt.print_level": self.print_level,
            "ipopt.max_iter": self.maxiter,
            "ipopt.tol": 1e-10,
            "print_time": False,
        }
        solver = ca.nlpsol("inverse_actuation", "ipopt", nlp, opts)
        if full_bounds:
            lbx = np.full(len(q_start), -np.pi)
            ubx = np.full(len(q_start), np.pi)
        else:
            lbx = np.maximum(-np.pi, q_start - self.max_step_rad)
            ubx = np.minimum(np.pi, q_start + self.max_step_rad)
        kwargs = {"x0": q_start, "lbx": lbx, "ubx": ubx}
        if g is not None:
            kwargs["lbg"] = lbg
            kwargs["ubg"] = ubg
        try:
            sol = solver(**kwargs)
            stats = solver.stats()
            x = np.asarray(sol["x"], dtype=float).reshape(-1)
            fun = float(sol["f"])
            return OptimizeResult(
                x=x,
                fun=fun,
                success=bool(stats.get("success", False)),
                message=str(stats.get("return_status", "")),
            )
        except Exception as exc:
            return OptimizeResult(x=np.asarray(q_start, dtype=float), fun=np.inf, success=False, message=str(exc))

    def _field_expr(self, p_m, q):
        ca = self.ca
        p = np.asarray(p_m, dtype=float).reshape(3)
        total = ca.SX.zeros(3, 1)
        for idx, center in enumerate(self.array.centers_m):
            moment = self._moment_expr(q, idx)
            r = ca.DM((p - center).reshape(3, 1))
            total += self._dipole_field_expr(r, moment)
        return total

    def _gradient_expr(self, p_m, q):
        ca = self.ca
        p_sym = ca.SX.sym("p", 3)
        b_expr = self._field_expr_symbolic_p(p_sym, q)
        jac = ca.jacobian(b_expr, p_sym)
        return ca.substitute(jac, p_sym, ca.DM(np.asarray(p_m, dtype=float).reshape(3)))

    def _field_expr_symbolic_p(self, p, q):
        ca = self.ca
        total = ca.SX.zeros(3, 1)
        for idx, center in enumerate(self.array.centers_m):
            rot = self._rotation_for_index_expr(q, idx)
            moment = self._moment_from_rotation_expr(rot, idx)
            r = p - ca.DM(center.reshape(3, 1))
            total += self._dipole_field_expr(r, moment)
            if self.array.include_hole_correction and self.model._hole_points is not None:
                total -= self._hole_field_expr(p, center, rot)
        return total

    def _moment_expr(self, q, idx):
        rot = self._rotation_for_index_expr(q, idx)
        return self._moment_from_rotation_expr(rot, idx)

    def _rotation_for_index_expr(self, q, idx):
        ca = self.ca
        gamma = q[2 * idx]
        beta = q[2 * idx + 1]
        phi = float(self.array.phi_rad[idx])
        return self._rotation_expr(phi, gamma, beta)

    def _moment_from_rotation_expr(self, rot, idx):
        ca = self.ca
        polarity = float(self.array.polarity_z[idx])
        local_z = ca.DM([0.0, 0.0, 1.0])
        return float(self.array.dipole_moment_magnitude) * polarity * ca.mtimes(rot, local_z)

    def _rotation_expr(self, phi, gamma, beta):
        ca = self.ca
        cp, sp = np.cos(phi), np.sin(phi)
        rz = ca.DM([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])
        cg, sg = ca.cos(gamma), ca.sin(gamma)
        cb, sb = ca.cos(beta), ca.sin(beta)
        rx = ca.vertcat(
            ca.horzcat(1.0, 0.0, 0.0),
            ca.horzcat(0.0, cb, -sb),
            ca.horzcat(0.0, sb, cb),
        )
        ry = ca.vertcat(
            ca.horzcat(cg, 0.0, sg),
            ca.horzcat(0.0, 1.0, 0.0),
            ca.horzcat(-sg, 0.0, cg),
        )
        return ca.mtimes([rz, rx, ry])

    def _dipole_field_expr(self, r, m):
        ca = self.ca
        norm = ca.sqrt(ca.dot(r, r) + self.model.min_distance_m ** 2)
        r_dot_m = ca.dot(r, m)
        return 1e-7 * (3.0 * r * r_dot_m / norm ** 5 - m / norm ** 3)

    def _hole_field_expr(self, p, center, rot):
        ca = self.ca
        rho = ca.mtimes(rot.T, p - ca.DM(center.reshape(3, 1)))
        local_density = ca.DM([0.0, 0.0, self.array.magnetization_a_per_m])
        local_total = ca.SX.zeros(3, 1)
        for xi in self.model._hole_points:
            moment = local_density * float(self.model._hole_dv)
            local_total += self._dipole_field_expr(rho - ca.DM(xi.reshape(3, 1)), moment)
        return ca.mtimes(rot, local_total)

    def _dipole_gradient_expr(self, r, m):
        ca = self.ca
        norm = ca.sqrt(ca.dot(r, r) + self.model.min_distance_m ** 2)
        r_dot_m = ca.dot(r, m)
        eye = ca.DM.eye(3)
        rr_t = ca.mtimes(r, r.T)
        rm_t = ca.mtimes(r, m.T)
        mr_t = ca.mtimes(m, r.T)
        term1 = 3.0 * (eye * r_dot_m + rm_t) / norm ** 5
        term2 = -15.0 * rr_t * r_dot_m / norm ** 7
        term3 = 3.0 * mr_t / norm ** 5
        return 1e-7 * (term1 + term2 + term3)
