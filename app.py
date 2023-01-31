#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import time
import math
from collections import Counter
from collections import deque
from PIL import ImageFont, ImageDraw, Image

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

from Picture_subject import picture_subject

buttons_in_start_scene = []
buttons_in_subject_hide_scene = []
buttons_in_subject_open_scene = []
buttons_in_playing_scene = []
buttons_in_judge_scene = []
buttons_in_result_scene = []
game_modes = ("start", "subject_hide", "subject_open",
              "playing", "judge", "dokidoki", "result")
game_mode = game_modes[0]
picture_subject_in_game = picture_subject()
white = (255, 255, 255)
text_color = (0, 0, 0)

paint_canvas_reset = True

sec = timer = 180  # タイマー秒数
timer_flag = False


class Pen:
    def __init__(self):
        self.color = (0, 0, 255)  # ペンの色(RGB)
        self.erase_color = (0, 0, 0)  # ペンの消しゴムの色(paint_canvasの背景と同じ色)
        self.thickness = 10  # ペンの太さ

    def setColor(self, color: tuple[int, int, int]) -> None:
        self.color = color

    def setThickness(self, thickness: int) -> None:
        self.thickness = thickness


class Button:
    def __init__(self, left_top_coord: tuple[int, int], right_bottom_coord: tuple[int, int], shape: str, color: tuple[int, int, int], thickness: int, func: any):
        self.left_top = left_top_coord
        self.right_bottom = right_bottom_coord
        self.shape = shape  # rectangle,circle
        self.color = color  # (int,int,int)
        self.thickness = thickness
        self.func = func


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=1920)
    parser.add_argument("--height", help='cap height', type=int, default=1080)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    global buttons_in_start_scene
    global buttons_in_subject_hide_scene
    global buttons_in_subject_open_scene
    global buttons_in_playing_scene
    global buttons_in_judge_scene
    global buttons_in_result_scene

    global game_modes
    global game_mode
    global picture_subject_in_game
    global paint_canvas_reset
    global timer
    global sec
    global timer_flag
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    print("cap_device == "+str(cap_device))
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # ラベル読み込み ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # 座標履歴 #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)
    hand_gesture_history = deque(maxlen=2)

    # フィンガージェスチャー履歴 ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0
    # デバッグモードとの切り替え用フラグ
    debugmode = False

    ret, image = cap.read()
    size = image.shape

    # タイマー変数 ############################################################

    previous_UNIX_time = 0  # 初期値0
    #########################################################################
    # UIのボタンの位置 一番左上が0,0 右下にいくにつれて大きくなる

    print(size)

    pen = Pen()  # ペンのインスタンス生成

    # test_button1 = Button((80,330),(180,430),(lambda x : x.setColor((255,0,0))))
    # test_button2 = Button((200,330),(300,430),(lambda x : x.setColor((0,0,255))))
    # buttons_in_game = [test_button1,test_button2]

    start_button = Button((400, 560), (860, 660), "rectangle",
                          (255, 255, 255), -1, (lambda x: change_gamemode(x)))

    buttons_in_start_scene = [start_button]
    show_subject_button = Button((400, 560), (860, 660), "rectangle",
                                 (255, 255, 255), -1, (lambda x: change_gamemode(x)))
    buttons_in_subject_hide_scene = [show_subject_button]
    confirm_subject_button = Button((400, 560), (860, 660), "rectangle",
                                    (255, 255, 255), -1, (lambda x: change_gamemode(x)))
    buttons_in_subject_open_scene = [confirm_subject_button]
    finish_button = Button((size[1]-140, 40), (size[1]-40, 140), "circle",
                           (255, 0, 0), -1, (lambda x: change_gamemode(x)))
    buttons_in_playing_scene = [finish_button]
    wrong_button = Button((0, 300), (200, size[0]), "rectangle",
                          (255, 0, 0), -1, (lambda x: change_gamemode(x)))
    correct_button = Button((size[1]-200, 300), (size[1], size[0]), "rectangle",
                            (0, 0, 255), -1, (lambda x: change_gamemode(x)))
    buttons_in_judge_scene = [wrong_button, correct_button]
    back_to_title_button = Button((400, 560), (860, 660), "rectangle",
                                  (255, 255, 255), -1, (lambda x: finish_game(x)))
    buttons_in_result_scene = [back_to_title_button]

    while True:
        if paint_canvas_reset:
            # 画像と同じサイズの黒で埋めた画像を用意
            paint_canvas = np.zeros(size, dtype=np.uint8)
            paint_canvas_reset = False
        fps = cvFpsCalc.get()

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        elif key & 0xFF == ord('r'):
            debugmode = not debugmode
        number, mode = select_mode(key, mode)

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # 外接矩形の計算
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # ランドマークの計算
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # 相対座標・正規化座標への変換
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # 学習データ保存
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # ハンドサイン分類
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                point_landmark = [0, 0]
                if hand_sign_id == 2:  # 指差しサイン
                    point_landmark = landmark_list[8]  # 人差指座標
                elif hand_sign_id == 1:  # グーの形のサイン
                    point_landmark = landmark_list[4]  # 親指の先
                elif hand_sign_id == 0:  # パーの形のサイン
                    point_landmark = landmark_list[9]  # 中指の付け根
                point_history.append(point_landmark)
                hand_gesture_history.append(hand_sign_id)

                print(point_landmark)

                # フィンガージェスチャー分類
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # 直近検出の中で最多のジェスチャーIDを算出
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # 描画
                if (hand_sign_id == 0):
                    pass  # パーだったら何もしない（ポインター的なものを表示する必要はあり）
                elif (hand_sign_id == 1):
                    if (hand_gesture_history[0] == 0):
                        process_menu(point_landmark, pen)
                    elif (hand_gesture_history[0] == 1):
                        paint_canvas = draw_latest_point_line(
                            paint_canvas, point_history, pen.thickness, pen.erase_color)
                    # cv.circle(paint_canvas,point_landmark,10,255,-1)#グーのときは消す（黒で線を描く）
                elif (hand_sign_id == 2):
                    if (hand_gesture_history[0] == 2):
                        paint_canvas = draw_latest_point_line(
                            paint_canvas, point_history, pen.thickness, pen.color)
                    # cv.circle(paint_canvas,point_landmark,10,0,-1)#指差しのときは白で線を描く

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        # タイマー処理 ######################################################

        current_UNIX_time = time.time()

        timer_flag, timer, previous_UNIX_time = timer_countdown(timer_flag, timer, current_UNIX_time,
                                                                previous_UNIX_time)
        if not timer_flag:
            timer = sec

        timer_str = str(timer)
        debug_image = draw_timer(debug_image, timer, (100, 100), 1)
        ###################################################################

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # paint_canvasをマスク画像に変換できるようにグレースケールにしてる

        paint_canvas2gray = cv.cvtColor(
            paint_canvas, cv.COLOR_BGR2GRAY)  # 黒背景に黒以外の色で線を描く
        ret, mask = cv.threshold(paint_canvas2gray, 1, 255, cv.THRESH_BINARY)
        mask_inv = cv.bitwise_not(mask)  # ビット反転して白背景にする

        image_src = cv.bitwise_and(
            image, image, mask=mask_inv)  # 画像に線の部分を黒抜きした画像
        paint_canvas = cv.bitwise_and(paint_canvas, paint_canvas, mask=mask)

        dst = cv.add(image_src, paint_canvas)

        game_image = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
        game_image = draw_info(game_image, fps, mode, number)

        # game_image = draw_UI_in_game(game_image)
        game_image = scene_transition(game_image)

        game_image = draw_cursor(game_image, point_history, history_length)
        # 画面反映 #############################################################
        # rキーで切り替えできる
        print(game_mode)
        if (debugmode):
            cv.imshow('Hand Gesture Recognition', debug_image)
        else:
            cv.imshow('Hand Gesture Recognition', game_image)
            # cv.imshow('Hand Gesture Recognition', paint_canvas)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def process_menu(coord, pen):
    global buttons_in_start_scene
    global buttons_in_playing_scene
    global buttons_in_judge_scene
    global buttons_in_result_scene
    global game_mode
    global game_modes
    global paint_canvas_reset
    print(game_mode)
    print(game_modes)
    if coord[0] != 0:  # 0,0以外の場所でのみ作用
        if game_mode == game_modes[0]:
            for button in buttons_in_start_scene:
                judge_coord(button, coord, 1)
        elif game_mode == game_modes[1]:
            for button in buttons_in_subject_hide_scene:
                judge_coord(button, coord, 2)
        elif game_mode == game_modes[2]:
            for button in buttons_in_subject_open_scene:
                judge_coord(button, coord, 3)
        elif game_mode == game_modes[3]:  # playing
            for button in buttons_in_playing_scene:
                if button.left_top[1] < 300:  # 強制終了ボタン
                    judge_coord(button, coord, 4)
                else:
                    judge_coord(button, coord, pen)
        elif game_mode == game_modes[4]:
            for button in buttons_in_judge_scene:
                judge_coord(button, coord, 5)
        elif game_mode == game_modes[5]:
            pass
        elif game_mode == game_modes[6]:
            for button in buttons_in_result_scene:
                judge_coord(button, coord, 0)


# 座標がボタンの座標内に存在するならば、ボタンの関数に第３引数を与えて実行する関数
def judge_coord(button: Button, coord: tuple[int, int], argument):
    if button.left_top[0] <= coord[0] <= button.right_bottom[0] and button.left_top[1] <= coord[1] <= button.right_bottom[1]:
        button.func(argument)

# あくまでテスト用


def draw_test_UI(image, buttons: Button):
    for button in buttons:
        cv.rectangle(image, button.left_top,
                     button.right_bottom, (255, 255, 255), -1)
    return image


def calc_circle_corner(center, radius):
    left_top = [0, 0]
    right_bottom = [0, 0]
    left_top[0] = center[0]-radius  # 左上の角のy
    left_top[1] = center[1]-radius  # 左上の角のx
    right_bottom[0] = center[0]+radius  # 右下の角のy
    right_bottom[1] = center[1]+radius  # 右下の角のx
    return left_top, right_bottom


def calc_circle_center_from_corners(left_top: tuple[int, int], right_bottom: tuple[int, int]):
    center_x = math.floor((left_top[0] + right_bottom[0])/2)
    center_y = math.floor((left_top[1] + right_bottom[1])/2)
    return (center_x, center_y)


def calc_cicle_radius_from_corners(left, right):
    radius = math.floor((left+right)/2)
    return radius


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # キーポイント
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # 相対座標に変換
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # 1次元リストに変換
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # 正規化
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # 相対座標に変換
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # 1次元リストに変換
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    # 接続線
    if len(landmark_point) > 0:
        # 親指
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # 人差指
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # 中指
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # 薬指
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # 小指
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # 手の平
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # キーポイント
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def timer_countdown(flag, timer, current_UNIX_time, previous_UNIX_time):
    if flag:
        if timer <= 0:  # タイマー終了時
            timer = 0
            previous_UNIX_time == 0  # 変数を元に戻す
            flag = False
        elif previous_UNIX_time == 0:  # タイマー起動後1度目のループ時
            previous_UNIX_time = time.time()
        else:
            pass

        if current_UNIX_time - previous_UNIX_time >= 1:
            timer = timer - 1
            previous_UNIX_time = time.time()

    return flag, timer, previous_UNIX_time


def draw_timer(image, timer, coord, font_size):
    cv.putText(image, str(timer), coord,
               cv.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 4, cv.LINE_AA)  # ここの引数を変えると色・場所・フォント等変更可

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_cursor(image, point_history, history_length):
    cursor_number = 5
    cursor_points = itertools.islice(
        point_history, history_length-cursor_number, None)
    for index, point in enumerate(cursor_points):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image

# point_historyの一番最後（最新の点）のみに点を描画する関数


def draw_latest_point(image, point_history, thickness, color):
    length = len(point_history)
    x = point_history[length-1][0]
    y = point_history[length-1][1]
    if x != 0 and y != 0:
        cv.circle(image, (x, y), thickness, color, -1)
    return image


def draw_latest_point_line(image, point_history, thickness, color):
    length = len(point_history)
    if (length >= 2):
        if (point_history[length-2] != [0, 0] and point_history[length-2] != point_history[length-1]):
            cv.line(image, (tuple(
                point_history[length-1])), tuple(point_history[length-2]), color, thickness)

    print(point_history[length-2], point_history[length-1])
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


def change_gamemode(number):
    global game_mode
    global game_modes
    game_mode = game_modes[number]


def finish_game(number):
    global paint_canvas_reset
    paint_canvas_reset = True
    change_gamemode(number)


def scene_transition(image):
    global buttons_in_start_scene
    global buttons_in_playing_scene
    global buttons_in_judge_scene
    global buttons_in_result_scene
    global game_modes
    global game_mode
    global sec
    global timer
    global timer_flag

    if game_mode == game_modes[0]:
        if not timer_flag:
            timer = 180
        return draw_UI_in_start_scene(image)
    elif game_mode == game_modes[1]:
        return draw_UI_in_subject_hide(image)
    elif game_mode == game_modes[2]:
        return draw_UI_in_subject_open(image)
    elif game_mode == game_modes[3]:
        if timer == 0:
            change_gamemode(4)
        elif not timer_flag:
            timer_flag = True
        else:
            pass
        return draw_UI_in_game(image)
    elif game_mode == game_modes[4]:
        if timer_flag:
            timer_flag = False
        return draw_UI_in_judge_scene(image)
    elif game_mode == game_modes[5]:
        if timer == 0:
            change_gamemode(6)
        elif not timer_flag:
            timer = 5
            timer_flag = True
        else:
            pass
        return draw_UI_in_dokidoki_scene(image)
    elif game_mode == game_modes[6]:
        if not timer_flag:
            timer_flag = False
        return draw_UI_in_result_scene(image)


def draw_buttons(image, buttons: Button):
    for button in buttons:
        if (button.shape == "rectangle"):
            cv.rectangle(image, button.left_top, button.right_bottom,
                         button.color, button.thickness)
        elif (button.shape == "circle"):
            center = calc_circle_center_from_corners(
                button.left_top, button.right_bottom)
            radius = calc_cicle_radius_from_corners(
                button.left_top[1], button.right_bottom[1])
            cv.circle(image, center, radius, button.color, button.thickness)
        else:
            print("button's shape is not designated")
            cv.rectangle(image, button.left_top, button.right_bottom,
                         button.color, button.thickness)

    return image


def draw_UI_background(image):
    # 透明化する図形を記述
    image2 = image.copy()
    cv.rectangle(image, (0, 500), (1920, 1080), (180, 180, 180), -1)
    weight = 0.5
    image3 = cv.addWeighted(image, weight, image2, 1-weight, -1)
    return image3


def draw_UI_in_start_scene(image):
    global buttons_in_start_scene
    image = draw_UI_background(image)
    image = draw_buttons(image, buttons_in_start_scene)
    image = putText_japanese(image, "ゲームをはじめる！", (400, 560), 50, (0, 0, 0))
    return image


def draw_UI_in_subject_hide(image):
    global buttons_in_subject_hide_scene
    image = draw_UI_background(image)
    image = draw_buttons(image, buttons_in_subject_hide_scene)
    image = putText_japanese(image, "ボタンをつかむとテーマが表示されます。",
                             (50, 250), 40, (0, 0, 0))
    image = putText_japanese(
        image, "お題を当てる人は画面を見ないでください。", (50, 300), 50, (0, 0, 0))
    return image


def draw_UI_in_subject_open(image):
    global buttons_in_subject_open_scene
    global picture_subject_in_game
    global text_color
    image = draw_UI_background(image)
    image = draw_buttons(image, buttons_in_subject_open_scene)
    image = putText_japanese(image, "お題は ", (50, 50), 40, text_color)
    image = putText_japanese(
        image, picture_subject_in_game, (50, 100), 80, text_color)
    image = putText_japanese(image, " です。", (50, 200), 40, text_color)
    image = putText_japanese(image, "ボタンをつかむとゲームが開始されます。",
                             (50, 250), 40, text_color)
    image = putText_japanese(
        image, "準備ができたらボタンをつかんでください。", (50, 300), 40, text_color)
    return image


def draw_UI_in_game(image):
    global buttons_in_playing_scene
    global timer
    image = draw_UI_background(image)
    image = draw_buttons(image, buttons_in_playing_scene)
    image = draw_timer(image, timer, (640, 50), 1)
    return image


def draw_UI_in_judge_scene(image):
    global buttons_in_judge_scene
    # image = draw_UI_background(image)
    image = draw_buttons(image, buttons_in_judge_scene)
    return image


def draw_UI_in_dokidoki_scene(image):
    global timer
    image = draw_timer(image, timer, (640, 360), 3)
    return image


def draw_UI_in_result_scene(image):
    global buttons_in_result_scene
    global text_color
    # image=draw_UI_background(image)
    image = putText_japanese(image, "お題は ", (50, 50), 40, text_color)
    image = putText_japanese(
        image, picture_subject_in_game, (50, 100), 80, text_color)
    image = putText_japanese(image, " でした。", (50, 200), 40, text_color)
    image = draw_buttons(image, buttons_in_result_scene)
    return image

# 日本語を描画する関数


def putText_japanese(image, text, point, size, color):
    # Notoフォントとする
    font = ImageFont.truetype("fonts/NotoSansJP-Light.otf", size)

    # imgをndarrayからPILに変換
    img_pil = Image.fromarray(image)

    # drawインスタンス生成
    draw = ImageDraw.Draw(img_pil)

    # テキスト描画
    draw.text(point, text, fill=color, font=font)

    # PILからndarrayに変換して返す
    return np.array(img_pil)


if __name__ == '__main__':
    main()
