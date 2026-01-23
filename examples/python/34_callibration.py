import rby1_sdk as rby
import numpy as np
import time



def ex4_move_to(right_arm, minimum_time=2.0):
    rc_builder = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            # Set Right Command with Joint Position Command
            .set_right_arm_command(
                rby.JointPositionCommandBuilder()
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(0.5))
                .set_minimum_time(minimum_time)
                .set_position(right_arm)
            )
        )
    )
    rc = robot.send_command(rc_builder).get()
    return rc.finish_code == rby.RobotCommandFeedback.FinishCode.Ok



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
            + (1/theta**2) * (1 - theta/(2*np.tan(theta/2))) * (w_hat @ w_hat)
        )
        v = A @ t

    return np.hstack([v, w])   # 6×1 twist


# ===============================
# Robot setup
# ===============================
robot = rby.create_robot_a("localhost:50051")
model = robot.model()

robot.connect()
robot.power_on(".*")
robot.servo_on(".*")
robot.enable_control_manager(False)

dyn_model = robot.get_dynamics()
RIGHT_ARM_IDX = model.right_arm_idx
ndof = len(RIGHT_ARM_IDX)

BASE, EE = 0, 1


# ===============================
# 1️⃣ 진짜 zero offset (시뮬에서만 사용)
# ===============================
q_offset_true = np.deg2rad([0.5, -1.0, 1.0, 0.5, -0.5, 0.5, -0.6])


# ===============================
# 2️⃣ 여러 개의 명령 pose
# ===============================
q_cmd_list = [
    np.deg2rad([-27.5, -38.3,  68.5, -56.9,  8.9, -69.1, -41.8]),
    np.deg2rad([-10.0, -50.0,  40.0, -70.0, 20.0, -30.0, -20.0]),
    np.deg2rad([ 20.0, -30.0,  60.0, -40.0, 10.0, -80.0, -10.0]),
    # np.deg2rad([-37.5, -18.3,  68.5, -56.9,  8.9, -69.1, -41.8]),
    # np.deg2rad([ 40.0, -70.0,  40.0, -70.0, 20.0, -30.0, -20.0]),
    # np.deg2rad([ 20.0, -30.0,  60.0, -40.0, 10.0, -60.0, -10.0]),
    # np.deg2rad([ 20.0, -30.0,  60.0, -40.0, 10.0, -80.0,  10.0]),
    # np.deg2rad([-37.5, -18.3,  28.5, -56.9, 18.9,  69.1,  41.8]),
    # np.deg2rad([ 40.0, -70.0,  40.0, -30.0, 20.0, -30.0, -20.0]),
    # np.deg2rad([ 20.0, -30.0,  60.0, -10.0, 10.0, -60.0, -10.0]),
]


# ===============================
# 3️⃣ Fake camera measurements
# ===============================
T_cam_list = []

for q_cmd in q_cmd_list:
    
    # ex4_move_to(q_cmd)
    q_full = robot.get_state().position.copy()
    q_full[RIGHT_ARM_IDX] = q_cmd + q_offset_true

    dyn_state = dyn_model.make_state(["base", "ee_right"], model.robot_joint_names)
    dyn_state.set_q(q_full)
    dyn_model.compute_forward_kinematics(dyn_state)

    T = dyn_model.compute_transformation(dyn_state, BASE, EE)
    # print("T_cam", T)
    # 작은 카메라 노이즈
    T[:3, 3] += np.random.normal(0, 0.002, 3)

    T_cam_list.append(T)
    # 실제 로봇에선 여기에 카메라 측정 T값을 넣어줘야함


# ===============================
# 4️⃣ Calibration (unknown offset)
# ===============================
q_offset = np.zeros(ndof)
MAX_ITER = 20

lambda2 = 1e-3

for it in range(MAX_ITER):
    H = np.zeros((ndof, ndof))
    g = np.zeros(ndof)

    for q_cmd, T_cam in zip(q_cmd_list, T_cam_list):
        q_full = robot.get_state().position.copy()
        q_full[RIGHT_ARM_IDX] = q_cmd + q_offset

        dyn_state = dyn_model.make_state(["base", "ee_right"], model.robot_joint_names)
        dyn_state.set_q(q_full)
        dyn_model.compute_forward_kinematics(dyn_state)

        T_fk = dyn_model.compute_transformation(dyn_state, BASE, EE)
        # print("T_fk", T_fk)
        T_err = np.linalg.inv(T_fk) @ T_cam
        xi = se3_log(T_err)

        J = dyn_model.compute_body_jacobian(dyn_state, BASE, EE)
        Jr = J[:, 7:14]

        H += Jr.T @ Jr
        g += Jr.T @ xi

    delta = -np.linalg.solve(H + lambda2 * np.eye(ndof), g)
    delta = np.clip(delta, -np.deg2rad(0.5), np.deg2rad(0.5))
    q_offset += delta

    print(f"[{it}] |δq| = {np.linalg.norm(delta)}")
    if np.linalg.norm(delta) < 1e-6:
        break


# ===============================
# 5️⃣ 결과
# ===============================
print("\nTrue offset     :", np.rad2deg(q_offset_true))
print("Estimated offset:", np.rad2deg(q_offset))
