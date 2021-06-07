import numpy as np


def vec_len(vector: np.ndarray) -> np.float64:
    return np.sqrt((vector ** 2).sum())


def rad_to_deg(rad: np.float64) -> np.float64:
    return rad * 180 / np.pi


def cos_compute(part1: np.ndarray, part2: np.ndarray) -> np.float64:
    return np.dot(part1, part2) / (vec_len(part1) * vec_len(part2))


def get_limb(parts: dict, kps: np.ndarray, limb_name: str) -> np.ndarray:
    limb = parts[limb_name]
    limb = np.array([kps[limb[0]], kps[limb[1]]])
    limb = limb.astype(np.float32)[:, :2]
    return limb


def angle_compute(parts: dict, kps: np.ndarray, limb1: str, limb2: str) -> np.float64:
    first_limb = get_limb(parts, kps, limb1)
    second_limb = get_limb(parts, kps, limb2)

    first = first_limb[1] - first_limb[0]
    second = second_limb[1] - second_limb[0]

    cosLimbs = cos_compute(first, second)
    return rad_to_deg(np.arccos(cosLimbs))


def is_look(kps: np.ndarray) -> str:
    if all(kps[3] > 0) and all(kps[4] > 0):
        return 'forward'
    if all(kps[3] > 0) and all(kps[1] > 0):
        return 'slightly_right'
    if all(kps[4] > 0) and all(kps[2] > 0):
        return 'slightly_left'
    if all(kps[3] > 0):
        return 'right'
    if all(kps[4] > 0):
        return 'left'
    return 'back'


def is_stand_straight(pose: dict):
    return 1 - np.mean(np.array(list(pose.values())) / 90)


def is_stand(pose: dict):
    return 1 - np.mean(np.array([pose['left_side_leg'], pose['right_side_leg'],
                                 pose['left_leg'], pose['right_leg']]) / 90)


def is_sit(pose: dict):
    return np.mean(np.array([pose['left_side_leg'], pose['right_side_leg'],
                             pose['left_leg'], pose['right_leg']]) / 90)


def is_arms_up(pose: dict):
    return np.mean((1 - (np.array([pose['left_arm'], pose['right_arm']]) / 180),
                    np.array([pose['left_side_arm'],
                              pose['right_side_arm']]) / 180))


def is_left_arm_up(pose: dict):
    return np.mean((1 - pose['left_arm'] / 180, pose['left_side_arm'] / 180))


def is_left_arm_down(pose: dict):
    return np.mean((1 - pose['left_arm'] / 180, 1 - pose['left_side_arm'] / 180))


def is_left_arm_side(pose: dict):
    return np.mean((1 - pose['left_arm'] / 180, 1/2 - pose['left_side_arm'] / 180))


def is_right_arm_up(pose: dict):
    return np.mean((1 - pose['right_arm'] / 180, pose['right_side_arm'] / 180))


def is_right_arm_down(pose: dict):
    return np.mean((1 - pose['right_arm'] / 180, 1 - pose['right_side_arm'] / 180))


def is_right_arm_side(pose: dict):
    return np.mean((1 - pose['right_arm'] / 180, 1/2 - pose['right_side_arm'] / 180))


class ActivityRecognition:
    PARTS = {
            'shoulders': (5, 6),
            'left_arm_start': (5, 7),
            'left_arm_end': (7, 9),
            'right_arm_start': (6, 8),
            'right_arm_end': (8, 10),
            'left_side': (5, 11),
            'right_side': (6, 12),
            'bottom': (11, 12),
            'left_leg_start': (11, 13),
            'left_leg_end': (13, 15),
            'right_leg_start': (12, 14),
            'right_leg_end': (14, 16)
        }

    def __init__(self, kps, threshold=0.7):
        self.kps = kps
        self.threshold = threshold
        self.PARTS = {
                'shoulders': (5, 6),
                'left_arm_start': (5, 7),
                'left_arm_end': (7, 9),
                'right_arm_start': (6, 8),
                'right_arm_end': (8, 10),
                'left_side': (5, 11),
                'right_side': (6, 12),
                'bottom': (11, 12),
                'left_leg_start': (11, 13),
                'left_leg_end': (13, 15),
                'right_leg_start': (12, 14),
                'right_leg_end': (14, 16)
            }

        self.pose_angles = {
            'right_arm': angle_compute(self.PARTS, kps, 'right_arm_start', 'right_arm_end'),
            'left_arm': angle_compute(self.PARTS, kps, 'left_arm_start', 'left_arm_end'),
            'right_leg': angle_compute(self.PARTS, kps, 'right_leg_start', 'right_leg_end'),
            'left_leg': angle_compute(self.PARTS, kps, 'left_leg_start', 'left_leg_end'),
            'right_side_arm': angle_compute(self.PARTS, kps, 'right_arm_start', 'right_side'),
            'left_side_arm': angle_compute(self.PARTS, kps, 'left_arm_start', 'left_side'),
            'right_side_leg': angle_compute(self.PARTS, kps, 'right_leg_start', 'right_side'),
            'left_side_leg': angle_compute(self.PARTS, kps, 'left_leg_start', 'left_side'),
        }

        self.ACTIVITIES = {
            'look_direction': self.look_direction,
            'stand': self.stand,
            'sit': self.sit,
            'arms': self.arms
        }

    def full_report(self):
        return [ func() for func in self.ACTIVITIES.values() ]

    def look_direction(self) -> str:
        return is_look(self.kps)

    def stand(self):
        return 'stand' if is_stand(self.pose_angles) > self.threshold else None

    def sit(self):
        return 'sit' if is_sit(self.pose_angles) > self.threshold else None

    def arms(self):
        a = is_right_arm_up(self.pose_angles)
        right_arm = 'up'
        if a < self.threshold:
            a = is_right_arm_down(self.pose_angles)
            right_arm = 'down'
        if a < self.threshold:
            a = is_right_arm_side(self.pose_angles)
            right_arm = 'side'

        a = is_left_arm_up(self.pose_angles)
        left_arm = 'up'
        if a < self.threshold:
            a = is_left_arm_down(self.pose_angles)
            left_arm = 'down'
        if a < self.threshold:
            a = is_left_arm_side(self.pose_angles)
            left_arm = 'side'

        return f'left {left_arm}, right {right_arm}'
