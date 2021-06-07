from pose import PoseRecognition
from poseedgetpu import PoseRecognition
from activity import ActivityRecognition

import cv2
import time
import sys
import numpy as np

def main(args):
    """
    main для проверки как все работает при программирование модуля
    """
    t_start = time.time()
    # net = PoseRecognition()
    # net = PoseRecognition(tflite_file='posenet_mobilenet_v1_075_353_481_quant_decoder.tflite')
    net = PoseRecognition(tflite_file='posenet_mobilenet_v1_075_353_481_quant_decoder_edgetpu.tflite')

    cap = cv2.VideoCapture(args[1])

    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = round(fps)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

    t_net_all = []
    t_act_all = []
    t_all = []
    i = 0

    #while True:
    for _ in range(300):
        t_start = time.time()
        i += 1
        if i % fps == 0:
            print(i, '/', total, 'frames')
        ret, frame = cap.read()

        t_net_start = time.time()
        kps = net.run(frame)
        t_net = time.time() - t_net_start

        t_act_start = time.time()
        # activity = ActivityRecognition(kps=kps)
        # activity.look_direction()
        # activity.full_report()
        
        t_act = time.time() - t_act_start
        
        t_all.append(time.time() - t_start)
        t_net_all.append(t_net)
        t_act_all.append(t_act)

        # frame = cv2.resize(frame, tuple(net.input_image_shape[:2]))
        if kps is not None:
            for k in kps:
                frame = draw_kps_lines(frame, k, ActivityRecognition.PARTS.values())
                x_m = int(k[:, 0].mean())
                y_m = int(k[:, 1].mean())
                activity = ActivityRecognition(kps=k)
                s = activity.stand()
                if s is None:
                    s = activity.sit()
                if s is None:
                    s = ""
                cv2.putText(frame, s, (y_m, x_m), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            # frame = draw_kps_lines(frame, kps, activity.PARTS.values())
        out.write(frame)
        #cv2.imshow('Img', frame)
        #if cv2.waitKey(25) & 0xFF == ord('q'):
        #    break

    # t_all = time.time() - t_start

    names = ['All:', 'Net full:', '- image:', '- network:', '- parse:', 'Activity:']
    # values = [t_all, t_net_all, t_net_image, t_net_net, t_net_parse, t_act_all]
    values = [t_all, t_net_all, net.time_img, net.time_net, net.time_parse, t_act_all]

    print('Overall:')
    print(*[ f'{i}\t{j}' for i, j in zip(names, np.sum(values, axis=1))], sep='\n')

    print('Normed (mean):')
    print(*[ f'{i}\t{j}' for i, j in zip(names, np.mean(values, axis=1))], sep='\n')

    print('Normed (median):')
    print(*[ f'{i}\t{j}' for i, j in zip(names, np.median(values, axis=1))], sep='\n')

    print('mean', *[ f'{i}' for i in np.mean(values, axis=1)], sep=',')
    print('median', *[ f'{i}' for i in np.median(values, axis=1)], sep=',')

    #     print(activity.look_direction())
    #     print(activity.stand())
    #     print(activity.sit())
    #     print(activity.arms())

    # image = draw_kps(image, kps)
    # cv2.imshow('Img', image)
    # cv2.waitKey()


def draw_kps(show_img, kps, ratio=None):
    for i in range(5, kps.shape[0]):
        if kps[i, 2]:
            if isinstance(ratio, tuple):
                cv2.circle(show_img, (int(round(kps[i, 1] * ratio[1])), int(
                    round(kps[i, 0] * ratio[0]))), 2, (0, 255, 255), round(int(1 * ratio[1])))
                continue
            cv2.circle(show_img, (int(kps[i, 1]), int(kps[i, 0])), 2, (0, 255, 255), -1)
    return show_img


def draw_kps_lines(image, kps, lines):
    image = draw_kps(image, kps)
    for i in lines:
        if not (kps[i[0], 2] and kps[i[1], 2]):
            continue
        cv2.line(image,
                 (int(kps[i[0], 1]), int(kps[i[0], 0])),
                 (int(kps[i[1], 1]), int(kps[i[1], 0])),
                 (0, 255, 255), 3)
    return image


if __name__ == '__main__':
    main(sys.argv)
