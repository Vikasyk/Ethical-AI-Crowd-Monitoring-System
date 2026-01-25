# SORT: Simple Online and Realtime Tracking
# Full working implementation

from __future__ import print_function
import numpy as np
from filterpy.kalman import KalmanFilter


def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    x1 = x[0] - w / 2.0
    y1 = x[1] - h / 2.0
    x2 = x[0] + w / 2.0
    y2 = x[1] + h / 2.0
    return np.array([x1, y1, x2, y2]).reshape((1, 4))


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R *= 0.01
        self.kf.P *= 10
        self.kf.Q *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0
        
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)


def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_gt[2]-bb_gt[0]) +
              (bb_test[3]-bb_test[1])*(bb_gt[3]-bb_gt[1]) - wh)
    return o


def associate_detections_to_trackers(dets, trks, iou_threshold=0.3):
    if len(trks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.array([])

    iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)

    for d in range(len(dets)):
        for t in range(len(trks)):
            iou_matrix[d, t] = iou(dets[d], trks[t])

    matched_indices = np.argwhere(iou_matrix > iou_threshold)

    unmatched_dets = np.setdiff1d(np.arange(len(dets)), matched_indices[:, 0])
    unmatched_trks = np.setdiff1d(np.arange(len(trks)), matched_indices[:, 1])

    matches = []
    for m in matched_indices:
        matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, unmatched_dets, unmatched_trks


class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.delete(trks, to_del, axis=0)
        self.trackers = [t for i, t in enumerate(self.trackers) if i not in to_del]

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)

        output = []
        for t in self.trackers:
            if (t.time_since_update < 1) and (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                output.append(np.append(t.get_state()[0], t.id))

        return np.array(output)