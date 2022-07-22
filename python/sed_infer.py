import os
import time
import yaml
import cv2
import re
import glob
import numpy as np
from collections import defaultdict
import paddle

from benchmark_utils import PaddleInferBenchmark
from preprocess import decode_image
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'])))
sys.path.insert(0, parent_path)

from det_infer import Detector, get_test_images, print_arguments, bench_log, PredictConfig, load_predictor
from mot_utils import argsparser, Timer, get_current_memory_mb, video2frames, _is_valid_video
from mot.tracker import JDETracker, DeepSORTTracker
from mot.utils import MOTTimer, write_mot_results, get_crops, clip_box, flow_statistic
from mot.visualize import plot_tracking, plot_tracking_dict,visualize_box_mask

from mot.mtmct.utils import parse_bias
from mot.mtmct.postprocess import trajectory_fusion, sub_cluster, gen_res, print_mtmct_result
from mot.mtmct.postprocess import get_mtmct_matching_results, save_mtmct_crops, save_mtmct_vis_results


class SDE_Detector(Detector):
    """
    Args:
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        tracker_config (str): tracker config path
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
        output_dir (string): The path of output, default as 'output'
        threshold (float): Score threshold of the detected bbox, default as 0.5
        save_images (bool): Whether to save visualization image results, default as False
        save_mot_txts (bool): Whether to save tracking results (txt), default as False
        draw_center_traj (bool): Whether drawing the trajectory of center, default as False
        secs_interval (int): The seconds interval to count after tracking, default as 10
        do_entrance_counting(bool): Whether counting the numbers of identifiers entering
            or getting out from the entrance, default as False，only support single class
            counting in MOT.
        reid_model_dir (str): reid model dir, default None for ByteTrack, but set for DeepSORT
        mtmct_dir (str): MTMCT dir, default None, set for doing MTMCT
    """

    def __init__(self,
                 model_dir,
                 tracker_config,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 output_dir='output',
                 threshold=0.5,
                 save_images=False,
                 save_mot_txts=False,
                 draw_center_traj=False,
                 secs_interval=10,
                 do_entrance_counting=False,
                 reid_model_dir=None,
                 mtmct_dir=None):
        super(SDE_Detector, self).__init__(
            model_dir=model_dir,
            device=device,
            run_mode=run_mode,
            batch_size=batch_size,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            output_dir=output_dir,
            threshold=threshold, )
        self.save_images = save_images
        self.save_mot_txts = save_mot_txts
        self.draw_center_traj = draw_center_traj
        self.secs_interval = secs_interval
        self.do_entrance_counting = do_entrance_counting

        assert batch_size == 1, "MOT model only supports batch_size=1."
        self.det_times = Timer(with_tracker=True)
        self.num_classes = len(self.pred_config.labels)

        # reid config
        self.use_reid = False if reid_model_dir is None else True
        if self.use_reid:
            self.reid_pred_config = self.set_config(reid_model_dir)
            self.reid_predictor, self.config = load_predictor(
                reid_model_dir,
                run_mode=run_mode,
                batch_size=50,  # reid_batch_size
                min_subgraph_size=self.reid_pred_config.min_subgraph_size,
                device=device,
                use_dynamic_shape=self.reid_pred_config.use_dynamic_shape,
                trt_min_shape=trt_min_shape,
                trt_max_shape=trt_max_shape,
                trt_opt_shape=trt_opt_shape,
                trt_calib_mode=trt_calib_mode,
                cpu_threads=cpu_threads,
                enable_mkldnn=enable_mkldnn)
        else:
            self.reid_pred_config = None
            self.reid_predictor = None

        assert tracker_config is not None, 'Note that tracker_config should be set.'
        self.tracker_config = tracker_config
        tracker_cfg = yaml.safe_load(open(self.tracker_config))
        cfg = tracker_cfg[tracker_cfg['type']]

        # tracker config
        self.use_deepsort_tracker = True if tracker_cfg[
            'type'] == 'DeepSORTTracker' else False
        if self.use_deepsort_tracker:
            # use DeepSORTTracker
            if self.reid_pred_config is not None and hasattr(
                    self.reid_pred_config, 'tracker'):
                cfg = self.reid_pred_config.tracker
            budget = cfg.get('budget', 100)
            max_age = cfg.get('max_age', 30)
            max_iou_distance = cfg.get('max_iou_distance', 0.7)
            matching_threshold = cfg.get('matching_threshold', 0.3)
            min_box_area = cfg.get('min_box_area', 0)
            vertical_ratio = cfg.get('vertical_ratio', 0)

            self.tracker = DeepSORTTracker(
                budget=budget,
                max_age=max_age,
                max_iou_distance=max_iou_distance,
                matching_threshold=matching_threshold,
                min_box_area=min_box_area,
                vertical_ratio=vertical_ratio, )
        else:
            # use ByteTracker
            use_byte = cfg.get('use_byte', False)
            det_thresh = cfg.get('det_thresh', 0.3)
            min_box_area = cfg.get('min_box_area', 0)
            vertical_ratio = cfg.get('vertical_ratio', 0)
            match_thres = cfg.get('match_thres', 0.9)
            conf_thres = cfg.get('conf_thres', 0.6)
            low_conf_thres = cfg.get('low_conf_thres', 0.1)

            self.tracker = JDETracker(
                use_byte=use_byte,
                det_thresh=det_thresh,
                num_classes=self.num_classes,
                min_box_area=min_box_area,
                vertical_ratio=vertical_ratio,
                match_thres=match_thres,
                conf_thres=conf_thres,
                low_conf_thres=low_conf_thres, )

        self.do_mtmct = False if mtmct_dir is None else True
        self.mtmct_dir = mtmct_dir

    def postprocess(self, inputs, result):
        # postprocess output of predictor
        np_boxes_num = result['boxes_num']

        if np_boxes_num[0] <= 0:
            print('[WARNNING] No object detected.')
            result = {'boxes': np.zeros([0, 6]), 'boxes_num': [0]}
        result = {k: v for k, v in result.items() if v is not None}
        return result

    def reidprocess(self, det_results, repeats=1):
        pred_dets = det_results['boxes']  # cls_id, score, x0, y0, x1, y1
        pred_xyxys = pred_dets[:, 2:6]

        ori_image = det_results['ori_image']


        ori_image_shape = ori_image.shape[:2]
        pred_xyxys, keep_idx = clip_box(pred_xyxys, ori_image_shape)


        if len(keep_idx[0]) == 0:
            det_results['boxes'] = np.zeros((1, 6), dtype=np.float32)
            det_results['embeddings'] = None
            return det_results

        pred_dets = pred_dets[keep_idx[0]]
        pred_xyxys = pred_dets[:, 2:6]

        w, h = self.tracker.input_size
        crops = get_crops(pred_xyxys, ori_image, w, h)

        # to keep fast speed, only use topk crops
        crops = crops[:len(keep_idx[0])]  # reid_batch_size
        det_results['crops'] = np.array(crops).astype('float32')
        det_results['boxes'] = pred_dets[:len(keep_idx[0])]

        input_names = self.reid_predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.reid_predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(det_results[input_names[i]])

        # model prediction
        for i in range(repeats):
            self.reid_predictor.run()
            output_names = self.reid_predictor.get_output_names()
            feature_tensor = self.reid_predictor.get_output_handle(output_names[
                0])
            pred_embs = feature_tensor.copy_to_cpu()

        det_results['embeddings'] = pred_embs
        return det_results

    def tracking(self, det_results):
        pred_dets = det_results['boxes']  # cls_id, score, x0, y0, x1, y1
        pred_embs = det_results.get('embeddings', None)

        if self.use_deepsort_tracker:
            # use DeepSORTTracker, only support singe class
            self.tracker.predict()
            online_targets = self.tracker.update(pred_dets, pred_embs)
            online_tlwhs, online_scores, online_ids = [], [], []
            if self.do_mtmct:
                online_tlbrs, online_feats = [], []
            for t in online_targets:
                if not t.is_confirmed() or t.time_since_update > 1:
                    continue
                tlwh = t.to_tlwh()
                tscore = t.score
                tid = t.track_id
                if self.tracker.vertical_ratio > 0 and tlwh[2] / tlwh[
                        3] > self.tracker.vertical_ratio:
                    continue
                online_tlwhs.append(tlwh)
                online_scores.append(tscore)
                online_ids.append(tid)
                if self.do_mtmct:
                    online_tlbrs.append(t.to_tlbr())
                    online_feats.append(t.feat)

            tracking_outs = {
                'online_tlwhs': online_tlwhs,
                'online_scores': online_scores,
                'online_ids': online_ids,
            }
            if self.do_mtmct:
                seq_name = det_results['seq_name']
                frame_id = det_results['frame_id']

                tracking_outs['feat_data'] = {}
                for _tlbr, _id, _feat in zip(online_tlbrs, online_ids,
                                             online_feats):
                    feat_data = {}
                    feat_data['bbox'] = _tlbr
                    feat_data['frame'] = f"{frame_id:06d}"
                    feat_data['id'] = _id
                    _imgname = f'{seq_name}_{_id}_{frame_id}.jpg'
                    feat_data['imgname'] = _imgname
                    feat_data['feat'] = _feat
                    tracking_outs['feat_data'].update({_imgname: feat_data})
            return tracking_outs
        else:
            # use ByteTracker, support multiple class
            online_tlwhs = defaultdict(list)
            online_scores = defaultdict(list)
            online_ids = defaultdict(list)
            if self.do_mtmct:
                online_tlbrs, online_feats = defaultdict(list), defaultdict(
                    list)
            online_targets_dict = self.tracker.update(pred_dets, pred_embs)
            for cls_id in range(self.num_classes):
                online_targets = online_targets_dict[cls_id]
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    tscore = t.score
                    if tlwh[2] * tlwh[3] <= self.tracker.min_box_area:
                        continue
                    if self.tracker.vertical_ratio > 0 and tlwh[2] / tlwh[
                            3] > self.tracker.vertical_ratio:
                        continue
                    online_tlwhs[cls_id].append(tlwh)
                    online_ids[cls_id].append(tid)
                    online_scores[cls_id].append(tscore)
                    if self.do_mtmct:
                        online_tlbrs[cls_id].append(t.tlbr)
                        online_feats[cls_id].append(t.curr_feat)

            if self.do_mtmct:
                assert self.num_classes == 1, 'MTMCT only support single class.'
                tracking_outs = {
                    'online_tlwhs': online_tlwhs[0],
                    'online_scores': online_scores[0],
                    'online_ids': online_ids[0],
                }
                seq_name = det_results['seq_name']
                frame_id = det_results['frame_id']
                tracking_outs['feat_data'] = {}
                for _tlbr, _id, _feat in zip(online_tlbrs[0], online_ids[0],
                                             online_feats[0]):
                    feat_data = {}
                    feat_data['bbox'] = _tlbr
                    feat_data['frame'] = f"{frame_id:06d}"
                    feat_data['id'] = _id
                    _imgname = f'{seq_name}_{_id}_{frame_id}.jpg'
                    feat_data['imgname'] = _imgname
                    feat_data['feat'] = _feat
                    tracking_outs['feat_data'].update({_imgname: feat_data})
                return tracking_outs

            else:
                tracking_outs = {
                    'online_tlwhs': online_tlwhs,
                    'online_scores': online_scores,
                    'online_ids': online_ids,
                }
                return tracking_outs

    def predict_image(self,
                      image_list,
                      run_benchmark=False,
                      repeats=1,
                      visual=True,
                      seq_name=None):
        num_classes = self.num_classes
        image_list.sort()
        ids2names = self.pred_config.labels
        #self.do_mtmct=False
        if self.do_mtmct:
            mot_features_dict = {}  # cid_tid_fid feats
        else:
            mot_results = []
        for frame_id, img_file in enumerate(image_list):
            if self.do_mtmct:
                if frame_id % 10 == 0:
                    print('Tracking frame: %d' % (frame_id))
            batch_image_list = [img_file]  # bs=1 in MOT model
            frame, _ = decode_image(img_file, {})
            if run_benchmark:
                # preprocess
                inputs = self.preprocess(batch_image_list)  # warmup
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                result_warmup = self.predict(repeats=repeats)  # warmup
                self.det_times.inference_time_s.start()
                result = self.predict(repeats=repeats)

                self.det_times.inference_time_s.end(repeats=repeats)

                # postprocess
                result_warmup = self.postprocess(inputs, result)  # warmup
                self.det_times.postprocess_time_s.start()
                det_result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()

                # tracking
                if self.use_reid:
                    det_result['frame_id'] = frame_id
                    det_result['seq_name'] = seq_name
                    det_result['ori_image'] = frame
                    det_result = self.reidprocess(det_result)
                result_warmup = self.tracking(det_result)
                self.det_times.tracking_time_s.start()
                if self.use_reid:
                    det_result = self.reidprocess(det_result)
                tracking_outs = self.tracking(det_result)
                self.det_times.tracking_time_s.end()
                self.det_times.img_num += 1

                cm, gm, gu = get_current_memory_mb()
                self.cpu_mem += cm
                self.gpu_mem += gm
                self.gpu_util += gu

            else:
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                self.det_times.inference_time_s.start()
                result = self.predict()
                self.det_times.inference_time_s.end()

                self.det_times.postprocess_time_s.start()
                det_result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()


                # tracking process
                self.det_times.tracking_time_s.start()
                if self.use_reid:
                    det_result['frame_id'] = frame_id
                    det_result['seq_name'] = seq_name
                    det_result['ori_image'] = frame
                    det_result = self.reidprocess(det_result)
                tracking_outs = self.tracking(det_result)
                self.det_times.tracking_time_s.end()
                self.det_times.img_num += 1

            online_tlwhs = tracking_outs['online_tlwhs']
            online_scores = tracking_outs['online_scores']
            online_ids = tracking_outs['online_ids']

            if self.do_mtmct:
                feat_data_dict = tracking_outs['feat_data']
                mot_features_dict = dict(mot_features_dict, **feat_data_dict)
            else:
                mot_results.append([online_tlwhs, online_scores, online_ids])

            if visual:
                if len(image_list) > 1 and frame_id % 10 == 0:
                    print('Tracking frame {}'.format(frame_id))
                frame, _ = decode_image(img_file, {})
                if isinstance(online_tlwhs, defaultdict):
                    im = plot_tracking_dict(
                        frame,
                        num_classes,
                        online_tlwhs,
                        online_ids,
                        online_scores,
                        frame_id=frame_id,
                        ids2names=[])
                else:
                    im = plot_tracking(
                        frame,
                        online_tlwhs,
                        online_ids,
                        online_scores,
                        frame_id=frame_id)
                save_dir = os.path.join(self.output_dir, seq_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(
                    os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), im)

        if self.do_mtmct:
            return mot_features_dict
        else:
            return mot_results

    def predict_video(self, video_file, camera_id):
        self.video_out_name = 'output.mp4'
        if camera_id != -1:
            self.capture = cv2.VideoCapture(camera_id)
        else:#视频用的
            self.capture = cv2.VideoCapture(video_file)
            video_out_name = os.path.split(video_file)[-1]
        # Get Video info : resolution, fps, frame count
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("fps: %d, frame_count: %d" % (self.fps, self.frame_count))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.out_path = os.path.join(self.output_dir, self.video_out_name)
        self.video_format = 'mp4v'
        self.fourcc = cv2.VideoWriter_fourcc(*self.video_format)
        self.writer = cv2.VideoWriter(self.out_path, self.fourcc, self.fps, (self.width, self.height))

        self.frame_id = 1
        self.timer = MOTTimer()
        self.results = defaultdict(list)
        num_classes = self.num_classes
        self.num_classes = 1
        self.data_type = 'mcmot' if num_classes > 1 else 'mot'
        self.ids2names = self.pred_config.labels

        self.center_traj = None
        self.entrance = None
        self.records = None
        if self.draw_center_traj:
            self.center_traj = [{} for i in range(self.num_classes)]
        if self.num_classes == 1:
            self.id_set = set()
            self.interval_id_set = set()
            self.in_id_list = list()
            self.out_id_list = list()
            self.prev_center = dict()
            self.records = list()
            self.entrance = [0, self.height / 2., self.width, self.height / 2.]
        self.video_fps = self.fps

        # while (1):
        #     ret, frame = capture.read()
        #     if not ret:
        #         break
        #     if self.frame_id % 10 == 0:
        #         print('Tracking frame: %d' % (self.frame_id))
        #     self.frame_id += 1
        #
        #     self.timer.tic()
        #     seq_name = self.video_out_name.split('.')[0]
        #     mot_results = self.predict_image(
        #         [frame], visual=False, seq_name=seq_name)
        #     self.timer.toc()
        #
        #     # bs=1 in MOT model
        #     online_tlwhs, online_scores, online_ids = mot_results[0]
        #
        #     # NOTE: just implement flow statistic for one class
        #     if num_classes == 1:
        #         result = (self.frame_id + 1, online_tlwhs, online_scores,
        #                   online_ids)
        #         statistic = flow_statistic(
        #             result, self.secs_interval, self.do_entrance_counting,
        #             self.video_fps, self.entrance, self.id_set, self.interval_id_set, self.in_id_list,
        #             self.out_id_list, self.prev_center, records, self.data_type, num_classes)
        #         records = statistic['records']
        #
        #     fps = 1. / self.timer.duration
        #     if self.use_deepsort_tracker:
        #         self.results[1].append(
        #             (self.frame_id + 1, online_tlwhs, online_scores, online_ids))
        #         ids2names = 'car'
        #         im = plot_tracking_dict(
        #             frame,
        #             num_classes,
        #             online_tlwhs,
        #             online_ids,
        #             online_scores,
        #             frame_id=self.frame_id,
        #             fps=fps,
        #             ids2names=ids2names,
        #             do_entrance_counting=self.do_entrance_counting,
        #             entrance=self.entrance,
        #             records=records,
        #             center_traj=self.center_traj)
        #
        #     self.writer.write(im)
        #     # if camera_id != -1:
        #     cv2.imshow('Mask Detection', im)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # if self.save_mot_txts:
        #     result_filename = os.path.join(
        #         self.output_dir, video_out_name.split('.')[-2] + '.txt')
        #     write_mot_results(result_filename, self.results)
        #
        #     result_filename = os.path.join(
        #         self.output_dir,
        #         video_out_name.split('.')[-2] + '_flow_statistic.txt')
        #     f = open(result_filename, 'w')
        #     for line in records:
        #         f.write(line)
        #     print('Flow statistic save in {}'.format(result_filename))
        #     f.close()
        #
        # self.writer.release()

    def predict_mtmct(self, mtmct_dir, mtmct_cfg):
        cameras_bias = mtmct_cfg['cameras_bias']
        cid_bias = parse_bias(cameras_bias)
        scene_cluster = list(cid_bias.keys())
        # 1.zone releated parameters
        use_zone = mtmct_cfg.get('use_zone', False)
        zone_path = mtmct_cfg.get('zone_path', None)

        # 2.tricks parameters, can be used for other mtmct dataset
        use_ff = mtmct_cfg.get('use_ff', False)
        use_rerank = mtmct_cfg.get('use_rerank', False)

        # 3.camera releated parameters
        use_camera = mtmct_cfg.get('use_camera', False)
        use_st_filter = mtmct_cfg.get('use_st_filter', False)

        # 4.zone releated parameters
        use_roi = mtmct_cfg.get('use_roi', False)
        roi_dir = mtmct_cfg.get('roi_dir', False)

        mot_list_breaks = []
        cid_tid_dict = dict()

        output_dir = self.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        seqs = os.listdir(mtmct_dir)
        for seq in sorted(seqs):
            fpath = os.path.join(mtmct_dir, seq)
            if os.path.isfile(fpath) and _is_valid_video(fpath):
                seq = seq.split('.')[-2]
                print('ffmpeg processing of video {}'.format(fpath))
                frames_path = video2frames(
                    video_path=fpath, outpath=mtmct_dir, frame_rate=25)
                fpath = os.path.join(mtmct_dir, seq)

            if os.path.isdir(fpath) == False:
                print('{} is not a image folder.'.format(fpath))
                continue
            if os.path.exists(os.path.join(fpath, 'img1')):
                fpath = os.path.join(fpath, 'img1')
            assert os.path.isdir(fpath), '{} should be a directory'.format(
                fpath)
            image_list = glob.glob(os.path.join(fpath, '*.jpg'))
            image_list.sort()
            assert len(image_list) > 0, '{} has no images.'.format(fpath)
            print('start tracking seq: {}'.format(seq))

            mot_features_dict = self.predict_image(
                image_list, visual=False, seq_name=seq)

            cid = int(re.sub('[a-z,A-Z]', "", seq))
            tid_data, mot_list_break = trajectory_fusion(
                mot_features_dict,
                cid,
                cid_bias,
                use_zone=use_zone,
                zone_path=zone_path)
            mot_list_breaks.append(mot_list_break)
            # single seq process
            for line in tid_data:
                tracklet = tid_data[line]
                tid = tracklet['tid']
                if (cid, tid) not in cid_tid_dict:
                    cid_tid_dict[(cid, tid)] = tracklet

        map_tid = sub_cluster(
            cid_tid_dict,
            scene_cluster,
            use_ff=use_ff,
            use_rerank=use_rerank,
            use_camera=use_camera,
            use_st_filter=use_st_filter)

        pred_mtmct_file = os.path.join(output_dir, 'mtmct_result.txt')
        if use_camera:
            gen_res(pred_mtmct_file, scene_cluster, map_tid, mot_list_breaks)
        else:
            gen_res(
                pred_mtmct_file,
                scene_cluster,
                map_tid,
                mot_list_breaks,
                use_roi=use_roi,
                roi_dir=roi_dir)

        camera_results, cid_tid_fid_res = get_mtmct_matching_results(
            pred_mtmct_file)

        crops_dir = os.path.join(output_dir, 'mtmct_crops')
        save_mtmct_crops(
            cid_tid_fid_res, images_dir=mtmct_dir, crops_dir=crops_dir)

        save_dir = os.path.join(output_dir, 'mtmct_vis')
        save_mtmct_vis_results(
            camera_results,
            images_dir=mtmct_dir,
            save_dir=save_dir,
            save_videos=False)

class test():

    def main(self,model_dir,tracker_config,cpu_threads,threshold,save_images,save_mot_txts,reid_model_dir,video_file,camera_id):
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        arch = yml_conf['arch']
        detector = SDE_Detector(
            model_dir,
            tracker_config=tracker_config,
            device='GPU',
            batch_size=1,
            cpu_threads=cpu_threads,
            threshold=threshold,
            save_images=save_images,
            save_mot_txts=save_mot_txts,
            draw_center_traj=True,
            reid_model_dir=reid_model_dir, )
        if video_file is not None or camera_id != -1:
            detector.predict_video(video_file, camera_id)
        while(1):
            ret, frame = detector.capture.read()
            if detector.frame_id % 10 == 0:
                print('Tracking frame: %d' % (detector.frame_id))
            detector.frame_id += 1
            detector.timer.tic()
            seq_name = detector.video_out_name.split('.')[0]
            mot_results = detector.predict_image(
                [frame], visual=False, seq_name=seq_name)
            detector.timer.toc()

            # bs=1 in MOT model
            online_tlwhs, online_scores, online_ids = mot_results[0]

            # NOTE: just implement flow statistic for one class
            if detector.num_classes == 1:
                result = (detector.frame_id + 1, online_tlwhs, online_scores,
                          online_ids)
                statistic = flow_statistic(
                    result, detector.secs_interval, detector.do_entrance_counting,
                    detector.video_fps, detector.entrance, detector.id_set, detector.interval_id_set, detector.in_id_list,
                    detector.out_id_list, detector.prev_center, detector.records, detector.data_type, detector.num_classes)
                records = statistic['records']

            fps = 1. / detector.timer.duration
            if detector.use_deepsort_tracker:
                detector.results[1].append(
                    (detector.frame_id + 1, online_tlwhs, online_scores, online_ids))
                ids2names = 'car'
                im = plot_tracking_dict(
                    frame,
                    detector.num_classes,
                    online_tlwhs,
                    online_ids,
                    online_scores,
                    frame_id=detector.frame_id,
                    fps=fps,
                    ids2names=ids2names,
                    do_entrance_counting=detector.do_entrance_counting,
                    entrance=detector.entrance,
                    records=records,
                    center_traj=detector.center_traj)

            detector.writer.write(im)
            # if camera_id != -1:
            cv2.imshow('Mask Detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if detector.save_mot_txts:
            result_filename = os.path.join(
                detector.output_dir, detector.video_out_name.split('.')[-2] + '.txt')
            write_mot_results(result_filename, detector.results)

            result_filename = os.path.join(
                detector.output_dir,
                detector.video_out_name.split('.')[-2] + '_flow_statistic.txt')
            f = open(result_filename, 'w')
            for line in records:
                f.write(line)
            print('Flow statistic save in {}'.format(result_filename))
            f.close()

        detector.writer.release()


if __name__ == '__main__':
    model_dir = r'E:\padde_Vehicle\Vehicle tracking\output_inference\ppyoloe_crn_l_80e_visdrone_largesize'
    tracker_config = r'E:\padde_Vehicle\Vehicle tracking\python\tracker_config.yml'
    reid_model_dir = r'E:\padde_Vehicle\Vehicle tracking\output_inference\deepsort_pplcnet_vehicle'
    cpu_threads = 16
    threshold = 0.6
    save_images = True
    save_mot_txts = True
    video_file = r'E:\padde_Vehicle\Vehicle tracking\video_file\demo.MP4'
    camera_id = -1
    Main = test()
    Main.main(model_dir,tracker_config,cpu_threads,threshold,save_images,save_mot_txts,reid_model_dir,video_file,camera_id)