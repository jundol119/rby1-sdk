import rby1_sdk as rby
import numpy as np
import time

# ===============================
# Lie algebra utils
# ===============================
def adjoint(T):
    R = T[:3,:3]
    p = T[:3,3]
    p_hat = np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])
    Ad = np.zeros((6,6))
    Ad[:3,:3] = R
    Ad[3:,3:] = R
    Ad[3:,:3] = p_hat @ R
    return Ad

def so3_exp(w):
    theta = np.linalg.norm(w)
    if theta < 1e-8:
        return np.eye(3)

    k = w / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

    return (
        np.eye(3)
        + np.sin(theta) * K
        + (1 - np.cos(theta)) * (K @ K)
    )


def se3_exp(xi):
    w = xi[:3]
    v = xi[3:]

    R = so3_exp(w)
    theta = np.linalg.norm(w)

    if theta < 1e-8:
        V = np.eye(3)
    else:
        K = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ]) / theta

        V = (
            np.eye(3)
            + (1 - np.cos(theta)) / theta * K
            + (theta - np.sin(theta)) / theta * (K @ K)
        )

    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = V @ v
    return T

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
q_offset_true = np.deg2rad([5, -5, 2, 5, -5, 5, 2])
xi_cam_true = np.array([0.02, -0.04, 0.03,   # rotation
                        0.01, 0.02, -0.015]) # translation

# expected position about camera braket
xi_cam_pose = np.array([0, 0, 0,   # rotation
                        0, 0, 0]) # translation
T_cam_pose = se3_exp(xi_cam_pose)
T_cam_true = se3_exp(xi_cam_true)

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

q_cmd_list = generate_random_q_list(n_samples=100)


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
    # T_cam_list.append(T_cam)
     
    T_meas = T_cam @ T_cam_pose @ T_cam_true 
    T_cam_list.append(T_meas)
# ===============================
# Gauss–Newton Offset Calibration
# ===============================
max_iter = 100
eps = 1e-3

q_offset = np.zeros(ndof)
xi_cam = np.zeros(6)

for it in range(max_iter):

    H = np.zeros((ndof+6, ndof+6))
    g = np.zeros(ndof+6)

    total_err = 0.0

    for q_cmd, T_meas in zip(q_cmd_list, T_cam_list):

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

        # ---- Camera extrinsic ----
        T_extrinsic = se3_exp(xi_cam)
        # ---- Full model ----
        T_model = T_fk @ T_cam_pose @ T_extrinsic
        
        
        # T_err = T_model @ np.linalg.inv(T_meas)
        # xi = se3_log(T_err)   # space error
        T_err =  T_meas @ np.linalg.inv(T_model) 
        xi = se3_log(T_err)   # space error

        Jb = dyn_model.compute_space_jacobian(dyn_state, BASE, EE) 
        
        # xi[3:] *= 0.1
        # J_joint[3:, :] *= 0.1
        # Jb[:, 7:][3:, :] *= 0.1
        # J_joint[:, 7:][3:, :] *= 0.1
        
        # Camera Jacobian = Identity
        J = np.zeros((6,13))
        J[:, :7] = Jb[:, RIGHT_ARM_IDX]
        # J[:, 7:] = np.eye(6)
        J[:, 7:] = adjoint(T_fk @ T_cam_pose)
        
        H += J.T @ J
        g += J.T @ xi
        total_err += np.linalg.norm(xi)

    
    dx = np.linalg.pinv(H) @ g
    q_offset += dx[:7]
    xi_cam += dx[7:]

    print(f"[Iter {it:02d}] |dq| = {np.linalg.norm(dx):.3e}, "
          f"|xi| = {total_err:.3e}")

    if np.linalg.norm(dx) < eps:
        print("Converged.")
        break


# ===============================
# Result
# ===============================
print("\n===== Joint Offset =====")
print("True offset [deg]:")
print(np.round(np.rad2deg(q_offset_true), 4))

print("Estimated offset [deg]:")
print(np.round(np.rad2deg(q_offset), 4))


print("\n===== Camera Extrinsic (xi) =====")
print("True xi_cam:")
print(np.round(xi_cam_true, 6))

print("Estimated xi_cam:")
print(np.round(xi_cam, 6))
