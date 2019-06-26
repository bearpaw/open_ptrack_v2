#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Simple Kinect rosbag parser to dump image, depthmap, and pointcloud
# Wei Yang (weiy@nvidia.com)
# Mar 14, 2019
# Usage
#   rosrun kinect2_playback kinect2_playback_ply.py
#   Then replay rosbag by
#   rosbag play --clock -r 0.1 seq_1.bag
#   Use -r to decrease the replay speed to allow data processing/dump time
#-------------------------------------------------------------------------------
import rospy
import rosbag
import message_filters as mf
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2

import os
import errno
import cv2
import time
import numpy as np
import argparse
import yaml
import json
import pypcd
import scipy.spatial.transform
from easydict import EasyDict as edict
from utils import Plotter3D, rot_x, rot_y, rot_z

def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def depth2pc(rgb, depth, xmap, ymap, intrinsics, mask=None):
    assert rgb.shape[0] == depth.shape[0] and rgb.shape[1] == depth.shape[1]
    pt2 = depth
    pt0 = (xmap - intrinsics[0][2]) * pt2 / intrinsics[0][0]
    pt1 = (ymap - intrinsics[1][2]) * pt2 / intrinsics[1][1]

    mask_depth = np.ma.getmaskarray(np.ma.masked_greater(pt2, 0))
    if mask is None:
        mask = mask_depth
    else:
        mask_semantic = np.ma.getmaskarray(np.ma.masked_equal(mask, 1))
        mask = mask_depth * mask_semantic

    choose = mask.flatten().nonzero()[0]

    pt2_valid = pt2.flatten()[choose][:, np.newaxis].astype(np.float32)
    pt0_valid = pt0.flatten()[choose][:, np.newaxis].astype(np.float32)
    pt1_valid = pt1.flatten()[choose][:, np.newaxis].astype(np.float32)
    b = rgb[:, :, 0].flatten()[choose][:, np.newaxis].astype(np.float32)
    g = rgb[:, :, 1].flatten()[choose][:, np.newaxis].astype(np.float32)
    r = rgb[:, :, 2].flatten()[choose][:, np.newaxis].astype(np.float32)

    ps_c = np.concatenate((pt0_valid, pt1_valid, pt2_valid, r, g, b), axis=1)

    return ps_c



def create_pcd_from_numpy(pc_np):
    pc_xyzrgb = pc_np.transpose().view(np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),
                      ('r', np.float32), ('g', np.float32), ('b', np.float32)]))
    pc_msg = pypcd.numpy_pc2.array_to_pointcloud2(pc_xyzrgb, merge_rgb=True) # sensor_msgs.msg._PointCloud2.PointCloud2
    pc = pypcd.PointCloud.from_msg(pc_msg)
    return pc

class KinectDecoder:
    def __init__(self, bagfile, devices, output,
            queue_size = 10,
            slop = 0.1,
            sync_method = 'exact',
            debug=False,
            vis=False
        ):
        self.output = output

        self.devices = devices
        self.num_devices = len(devices)
        self.num_subs = 4
        self.counter = 0
        self.debug = debug
        self.vis = vis

        # initialize node
        rospy.init_node('KinectDecoder', anonymous=True)

        # initialize Subscriber and TimeSynchronizer
        topic_list = []
        for k, v in devices.items():
            t = [v['translation'][axis] for axis in ['x', 'y', 'z']]
            quat = [v['rotation'][axis] for axis in ['x', 'y', 'z', 'w']]
            R = scipy.spatial.transform.Rotation.from_quat(quat).as_dcm()
            devices[k]['R']= R
            devices[k]['t']= np.array(t) * 1000
            topic_list.extend(
                ['/'+ k + '/qhd/camera_info',
                '/'+ k + '/qhd/image_color_rect/compressed',
                '/'+ k + '/qhd/image_depth_rect/compressed',]
            )
        sync_list = [mf.SimpleFilter() for topic in topic_list]
        if sync_method == 'exact':
            ts = mf.TimeSynchronizer(sync_list, queue_size)
        elif sync_method == 'approximate':
            ts = mf.ApproximateTimeSynchronizer(sync_list, queue_size, slop)
        else:
            rospy.logfatal('Unrecognized sync method: {}'.format(sync_method))
            return

        ts.registerCallback(self._sync_messages)
        # rospy.spin()

        # read bagfile
        self.synced_msgs = [[] for t in topic_list]

        print('Loading {} '.format(bagfile))
        bag = rosbag.Bag(bagfile)

        all_topics = bag.get_type_and_topic_info()
        print('--------\n All topics \n')
        print(all_topics[1].keys())
        print('--------\n')

        for topic, msg, stamp in bag.read_messages(topics=topic_list):
            sync_list[topic_list.index(topic)].signalMessage(msg)

        seq_len = len(self.synced_msgs[0])
        im_width, im_height = None, None
        for counter in range(seq_len):
            pre_vis = None
            end = time.time()

            pc_list = []
            for k, v in devices.items():
                # generate file names
                name = k
                camera_info = '/'+ k + '/qhd/camera_info'
                color = '/'+ k + '/qhd/image_color_rect/compressed'
                depth = '/'+ k + '/qhd/image_depth_rect/compressed'

                mkdir_p(os.path.join(self.output, name, 'camera_info'))
                mkdir_p(os.path.join(self.output, name, 'color'))
                mkdir_p(os.path.join(self.output, name, 'depth'))
                mkdir_p(os.path.join(self.output, name, 'pointcloud'))

                camera_path = os.path.join(self.output, name, 'camera_info', 'f_%.5d.json' % counter)
                color_path = os.path.join(self.output, name, 'color', 'f_%.5d.png' % counter)
                depth_path = os.path.join(self.output, name, 'depth', 'f_%.5d.png' % counter)
                pc_path = os.path.join(self.output, name, 'pointcloud', 'f_%.5d.pcd' % counter)

                # decode camera_info/image/depth/
                camera_info = self._decode_camerainfo(
                    self.synced_msgs[
                        topic_list.index(camera_info)
                    ][counter]
                )
                color_im = self._decode_image_color_compressed(
                    self.synced_msgs[
                        topic_list.index(color)
                    ][counter]
                )
                depth_im = self._decode_image_depth_compressed(
                    self.synced_msgs[
                        topic_list.index(depth)
                    ][counter]
                )

                # decode pointcloud
                if im_width == None:
                    im_height, im_width = depth_im.shape[0], depth_im.shape[1]
                    ymap = np.array([[y for x in range(im_width)] for y in range(im_height)])
                    xmap = np.array([[x for x in range(im_width)] for y in range(im_height)])

                pc_np = depth2pc(color_im, depth_im, xmap, ymap, camera_info['K'])
                pc = create_pcd_from_numpy(pc_np)
                pc.save_pcd(pc_path, compression='binary_compressed')

                # save to disc
                with open(camera_path, 'w') as outfile:
                    json.dump(camera_info, outfile)
                cv2.imwrite(color_path, color_im)
                cv2.imwrite(depth_path, depth_im)

                # Transform
                R = v['R']
                t = v['t']
                pc_np[:, 0:3] = np.matmul(pc_np[:, 0:3], R.transpose()) + t

                pc_list.append(pc_np)

                if self.debug:
                    depth_vis = self._show_image_depth(depth_im)
                    vis_image = np.concatenate((color_im, depth_vis), axis=1)
                    if pre_vis is None:
                        pre_vis = vis_image
                    else:
                        pre_vis = np.concatenate((pre_vis, vis_image), axis=0)

            # merge pt
            pc_np = np.concatenate(pc_list, axis=0)

            # rotate matrix
            global_R = rot_x(-90 * np.pi / 180)
            pc_np[:, 0:3] = np.matmul(pc_np[:, 0:3], global_R)

            mkdir_p(os.path.join(self.output, 'pointcloud_merge'))
            pc = create_pcd_from_numpy(pc_np)
            pc_path = os.path.join(self.output, 'pointcloud_merge', 'f_%.5d.pcd' % counter)
            pc.save_pcd(pc_path, compression='binary_compressed')

            if self.vis:
                plotter = Plotter3D()
                idx = np.random.permutation(pc_np.shape[0])[:100000]
                plotter.draw_points(pc_np[idx, 0:3], rgb=pc_np[idx, 3:], scale_factor=10)
                plotter.draw_axis(pc_np[idx, 0:3])
                plotter.show()


            if self.debug:
                mkdir_p(os.path.join(self.output, 'visualize'))
                pre_vis_path = os.path.join(self.output, 'visualize', 'f_%.5d.jpg' % counter)
                cv2.imwrite(pre_vis_path, pre_vis)
            end = time.time() - end
            rospy.loginfo('Decoding image %d/%d (%.4f sec)' % (counter + 1, seq_len, end))





    def _sync_messages(self, *msg_arg):
        for i, msg in enumerate(msg_arg):
            self.synced_msgs[i].append(msg)

    def _pose_processing(self, pc):
        return pc

    def _decode_image_color_compressed(self, data):
        im = np.frombuffer(data.data, dtype=np.uint8)
        return cv2.imdecode(im, cv2.IMREAD_COLOR)


    def _decode_image_depth_compressed(self, data):
        im = np.frombuffer(data.data, dtype=np.uint16) # milimeter
        im = cv2.imdecode(im, cv2.IMREAD_ANYDEPTH)
        return im

    def _decode_pointcloud(self, data):
        pc = pypcd.PointCloud.from_msg(data)
        return pc

    def _decode_camerainfo(self, data):
        return {
            'height': data.height,
            'width': data.width,
            'k': np.array(data.D[0:2]).tolist(),
            't': np.array(data.D[2:]).tolist(),
            'K': np.array(data.K).reshape((3, 3)).tolist(),
            'R': np.array(data.R).reshape((3, 3)).tolist(),
            'P': np.array(data.P).reshape((3, 4)).tolist()
        }

    def _show_image_depth(self, depth):
        tmp = depth.copy()
        tmp = tmp.astype(np.float32) / tmp.max() * 255
        im_color = cv2.applyColorMap(tmp.astype(np.uint8), cv2.COLORMAP_BONE)
        return im_color



def parse_args():
    parser = argparse.ArgumentParser(description='Save color/depth/pointcloud from bag')
    parser.add_argument('--camera_pose', help = 'camera pose yaml', default = '', type = str)
    parser.add_argument('--bag', help = 'bag file', default = '/home/weiy/openptrack_ws/test.bag', type = str)
    parser.add_argument('--output', help = 'output dir', default = '/home/weiy/tmp/test_bag', type = str)
    parser.add_argument('--debug', help = 'save visualization', action='store_true')
    parser.add_argument('--vis', help = 'visualize merged PointCloud', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.camera_pose) as f:
        dict = edict(yaml.load(f))

        bagfile = args.bag
        output_dir = args.output
        # import pdb; pdb.set_trace()
        devices = dict['poses']

        decoder = KinectDecoder(
            bagfile,
            devices,
            output_dir,
            sync_method='approximate',
            debug=args.debug,
            queue_size=100,
            slop=1,
            vis=args.vis
        )
