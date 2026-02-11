import rby1_sdk as rby
import numpy as np
import time

# ===============================
# Lie algebra utils
# ===============================
def so3_log(R):
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3)

    w_hat = (R - R.T) / (2 * np.sin(theta))
    return theta * np.array([
        w_hat[2,1],
        w_hat[0,2],
        w_hat[1,0]
    ])


def se3_log(T):
    R = T[:3, :3]
    t = T[:3, 3]

    w = so3_log(R)
    theta = np.linalg.norm(w)

    if theta < 1e-8:
        v = t
    else:
        w_hat = np.array([
            [    0, -w[2],  w[1]],
            [ w[2],     0, -w[0]],
            [-w[1],  w[0],     0]
        ]) / theta

        A = (
            np.eye(3)
            - 0.5 * w_hat
            + (1/theta**2) * (1 - theta/(2*np.tan(theta/2)))
            * (w_hat @ w_hat)
        )
        v = A @ t

    return np.hstack([w, v])   # (6,)


# ===============================
# Robot setup
# ===============================
robot = rby.create_robot_a("localhost:50051")
model = robot.model()

robot.connect()
robot.power_on(".*")
robot.servo_on(".*")
robot.reset_fault_control_manager()
robot.enable_control_manager(False)

dyn_model = robot.get_dynamics()

RIGHT_ARM_IDX = model.right_arm_idx[:7]
ndof = len(RIGHT_ARM_IDX)

BASE, EE = 0, 1


# ===============================
# Ground truth offset (simulation)
# ===============================
# q_offset_true = np.deg2rad([0.5, -1.0, 1.0, 0.5, -5.0, 0.5, 0.2])
q_offset_true = np.deg2rad([50, -10, 10, 5, -50, 5, 2])


# ===============================
# Command poses
# ===============================
joint_limits = np.array([
    [-2.0,  2.0],
    [-2.5,  0.0],
    [-1.5,  1.5],
    [-2.5,  0.0],
    [-3.141592654,  3.141592654],
    [-1.570796327,  1.570796327],
    [-1.570796327,  1.570796327]
])

def generate_random_q_list(n_samples=10, margin_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    q_list = []
    for _ in range(n_samples):
        q = []
        for lo, hi in joint_limits:
            span = hi - lo
            q.append(rng.uniform(lo + margin_ratio*span,
                                 hi - margin_ratio*span))
        q_list.append(np.array(q))
    return q_list

q_cmd_list = generate_random_q_list(n_samples=15)


# ===============================
# Nominal configuration
# ===============================
q_nominal = robot.get_state().position.copy()


# ===============================
# Fake camera measurements
# ===============================
T_cam_list = []

for q_cmd in q_cmd_list:
    q_full = q_nominal.copy()
    q_full[RIGHT_ARM_IDX] = q_cmd + q_offset_true

    dyn_state = dyn_model.make_state(
        ["link_torso_5", "ee_right"],
        model.robot_joint_names
    )
    dyn_state.set_q(q_full)
    dyn_model.compute_forward_kinematics(dyn_state)

    T_cam = dyn_model.compute_transformation(dyn_state, BASE, EE)
    T_cam_list.append(T_cam)


# ===============================
# Gauss–Newton Offset Calibration
# ===============================
max_iter = 100
eps = 1e-3

q_offset = np.zeros(ndof)

for it in range(max_iter):

    H = np.zeros((ndof, ndof))
    g = np.zeros(ndof)

    total_err = 0.0

    for q_cmd, T_cam in zip(q_cmd_list, T_cam_list):

        # 🔁 재선형화 지점
        q_full = q_nominal.copy()
        q_full[RIGHT_ARM_IDX] = q_cmd + q_offset

        dyn_state = dyn_model.make_state(
            ["link_torso_5", "ee_right"],
            model.robot_joint_names
        )
        dyn_state.set_q(q_full)
        dyn_model.compute_forward_kinematics(dyn_state)
        dyn_model.compute_diff_forward_kinematics(dyn_state)

        T_fk = dyn_model.compute_transformation(dyn_state, BASE, EE)

        # SE(3) body error
        # T_err = np.linalg.inv(T_fk) @ T_cam
        # xi = se3_log(T_err)
        T_err = T_cam @ np.linalg.inv(T_fk)
        xi = se3_log(T_err)   # space error

        Jb = dyn_model.compute_space_jacobian(dyn_state, BASE, EE)
        Jr = Jb[:, RIGHT_ARM_IDX]
        xi[3:] *= 0.1
        Jr[3:, :] *= 0.1

        H += Jr.T @ Jr
        g += Jr.T @ xi
        total_err += np.linalg.norm(xi)

    dq = np.linalg.pinv(H) @ g
    q_offset += dq

    print(f"[Iter {it:02d}] |dq| = {np.linalg.norm(dq):.3e}, "
          f"|xi| = {total_err:.3e}")

    if np.linalg.norm(dq) < eps:
        print("Converged.")
        break


# ===============================
# Result
# ===============================
print("\nTrue offset [deg]:     ",
      np.round(np.rad2deg(q_offset_true), 4))
print("Estimated offset [deg]:",
      np.round(np.rad2deg(q_offset), 4))
