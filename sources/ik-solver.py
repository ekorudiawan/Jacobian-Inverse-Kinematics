from os import link
import numpy as np

links = [30,20]

def Rx(theta):
    return np.array([[1, 0, 0, 0], 
                     [0, np.cos(theta), -np.sin(theta), 0], 
                     [0, np.sin(theta), np.cos(theta), 0],
                     [0, 0, 0, 1]])   

def Ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta), 0], 
                     [0, 1, 0, 0], 
                     [-np.sin(theta), 0, np.cos(theta), 0],
                     [0, 0, 0, 1]])

def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0, 0], 
                     [np.sin(theta), np.cos(theta), 0, 0], 
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def Translate(dx=0, dy=0, dz=0):
    return np.array([[1, 0, 0, dx],
                     [0, 1, 0, dy],
                     [0, 0, 1, dz],
                     [0, 0, 0, 1]])

def rotation_matrix_to_euler(R):
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        roll = np.arctan2(R[2,1] , R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
    else :
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0
    return roll, pitch, yaw

def transformation_matrix_to_xyzrpy(TF):
    x = TF[0,3]
    y = TF[1,3]
    z = TF[2,3]
    roll, pitch, yaw = rotation_matrix_to_euler(TF[0:3,0:3])
    return x, y, z, roll, pitch, yaw

def forward_kinematics(thetas):
    R1 = Rz(thetas[0])
    t1 = Translate(dx=links[0], dy=0, dz=0)
    R2 = Rz(thetas[1])
    t2 = Translate(dx=links[1], dy=0, dz=0)
    return R1.dot(t1).dot(R2).dot(t2)

def jacobian_matrix(thetas, eps=0.01):
    J = np.zeros((3, len(thetas)))
    for i in range(len(thetas)):
        _0T2_current = forward_kinematics(thetas)
        x_current, y_current, z_current, _, _, _ = transformation_matrix_to_xyzrpy(_0T2_current)
        thetas[i] += eps
        _0T2_next = forward_kinematics(thetas)
        x_next, y_next, z_next, _, _, _ = transformation_matrix_to_xyzrpy(_0T2_next)
        J[0,i] = (x_next - x_current) / eps
        J[1,i] = (y_next - y_current) / eps
        J[2,i] = (z_next - z_current) / eps
        thetas[i] -= eps
    return J

def inverse_kinematics(debug=True, x_target=0, y_target=0, z_target=0, max_iter=100, alpha=0.001):
    thetas = np.random.rand(2)
    J = jacobian_matrix(thetas=thetas)
    J_plus = np.linalg.pinv(J)
    for i in range(max_iter):
        _0T2 = forward_kinematics(thetas=thetas)
        x_current, y_current, z_current, _, _, _ = transformation_matrix_to_xyzrpy(_0T2)
        delta_pose = np.array([x_target-x_current, y_target-y_current, z_target-z_current])
        delta_thetas = J_plus.dot(delta_pose)
        thetas += alpha * delta_thetas
        if debug:
            print("Iter : ", i)
            print("Thetas : ", thetas)
            print("-------------")
    return thetas

def calc_pose_error(target_pose, result_pose):
    print("Target Pose :", target_pose)
    print("Result Pose :", result_pose)
    target_pose_transpose = target_pose.T
    position_error = (target_pose - result_pose)[0:3,3]
    dx = position_error[0]
    dy = position_error[1]
    dz = position_error[2]
    print("Position Error :", dx, dy, dz)
    rotation_error = target_pose_transpose.dot(result_pose)[0:3,0:3]
    droll, dpitch, dyaw = rotation_matrix_to_euler(rotation_error)
    print("Rotation Error : ", droll, dpitch, dyaw)
    mse = np.sqrt(dx**2 + dy**2 + dz**2 + droll**2 + dpitch**2 + dyaw**2)
    return mse

def main():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    random_thetas = np.random.rand(2)
    target_pose = forward_kinematics(thetas=random_thetas)
    target_x, target_y, _, _, _, _ = transformation_matrix_to_xyzrpy(target_pose)
    print("Target (x,y) : ", target_x, target_y)
    thetas = inverse_kinematics(debug=False, x_target=target_x, y_target=target_y, max_iter=500, alpha=0.1)
    print(thetas)
    result_pose = forward_kinematics(thetas=thetas)
    result_x, result_y, _, _, _, _ = transformation_matrix_to_xyzrpy(result_pose)
    print("Result (x,y) : ", result_x, result_y)
    error = calc_pose_error(target_pose, result_pose)
    print("Error : ", error)
    if error > 1:
        print("Fail")

if __name__ == "__main__":
    main()