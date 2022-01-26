# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import tensorflow as tf
import numpy as np

from typing import List, Optional, Tuple
from . import interpolate

from ..parser import graphics_tf_module
from ..models import stylegan

# TODO: Support TF2.0


def voxel_visibility_from_view(mask_voxel: tf.Tensor,
                               cam_pose: tf.Tensor,
                               ) -> tf.Tensor:
    """

    Args:
        mask_voxel: an `int32` or `int64` Tensor with shape [b, h, w, d] to mark the existence of voxel. 1 for
            non-empty, otherwise 0.
        cam_pose: a `float32` Tensor with shape [b, 3] for the camera position in voxel coordinate space

    Returns:
        an `int32` Tensor with shape [b, h, w, d] to mark the visibility of voxel from view. 0 for occluded, 1 for
            surface, 2 for empty but non-occluded.
    """
    return graphics_tf_module.voxel_visibility_from_view(mask_voxel, cam_pose)


def importance_coverage_sampling(label_ref: tf.Tensor,
                                 logits: tf.Tensor,
                                 n: int,
                                 sample_shape: Optional[List] = None,
                                 k: int = 5,
                                 d: float = 0.5,
                                 beta: float = 0.75,
                                 seed: int = None,
                                 name: Optional[str] = None) -> tf.Tensor:
    """
    The sampling strategy is first proposed in Kaiming's technical report "PointRend: Image Segmentation as Rendering".
        The sampling procedure could be divided into three steps: 1) over-sampling, to sample N times points than
        excepted, 2) importance-sampling, to rank all points by specified rules and select a partial set of points with
        high ranking and 3) coverage-sampling, to fulfill the rest of points from low ranking points.

    Args:
        label_ref: an `int32` Tensor, the label map to select probability from logits
        logits: a `float32` Tensor, the logits to provide the ranking score
        n: the total sampled points
        sample_shape: an int list, the shape to sample, default to None will infer from `logits`
        k: over-sampling times
        d: mean probability to compute the distance
        beta: the blending factor to balance the number of points between importance and uniform sets
        seed: random seed of uniform sampling
        name: operation name

    Returns:
        a `float32` Tensor to record the sampled points
    """
    label_ref = tf.compat.v1.convert_to_tensor(label_ref)
    logits = tf.compat.v1.convert_to_tensor(logits)
    if sample_shape is None:
        sample_shape = logits.shape[1:-1].as_list()

    def _individual_ranking(in_vars):
        score, pts = in_vars
        with tf.compat.v1.name_scope(None, default_name='Ranking', values=[score, pts]):
            sorted_idx = tf.argsort(tf.abs(score - d))
            sorted_pts = tf.gather(pts, sorted_idx)
            p1_num = int(n * beta)
            p2_num = n - p1_num
            p1_pts = sorted_pts[:p1_num, ...]
            p2_pts = tf.random.shuffle(sorted_pts[p1_num:, ...], seed=seed)[:p2_num]
            return tf.concat([p1_pts, p2_pts], axis=0)

    with tf.compat.v1.name_scope(name, default_name='ImportanceCoverageSampling', values=[label_ref, logits]):
        label_ref = tf.stop_gradient(label_ref)
        logits = tf.stop_gradient(logits)
        kn = tf.random.uniform(shape=[logits.shape[0].value, k * n, len(sample_shape)], dtype=tf.float32, seed=seed)
        kn *= sample_shape
        batch_index = np.arange(logits.shape[0].value)[..., np.newaxis, np.newaxis]
        batch_index = tf.convert_to_tensor(np.tile(batch_index, [1, kn.shape[1].value, 1]), dtype=tf.float32)
        interp_index = tf.reshape(tf.concat([batch_index, kn], axis=-1), [-1, len(sample_shape) + 1])

        interp_logit = interpolate.linear_nd_interpolate(logits, interp_index)
        interp_label = tf.argmax(interpolate.linear_nd_interpolate(label_ref, interp_index), axis=-1)
        interp_label = tf.stack([np.arange(interp_label.shape[0].value), interp_label], axis=-1)
        interp_score = tf.gather_nd(interp_logit, interp_label)
        interp_score = tf.reshape(interp_score, [logits.shape[0].value, k * n])

        sampled_pts = tf.map_fn(_individual_ranking, (interp_score, kn), dtype=tf.float32)
        return sampled_pts


def iterative_farthest_point_sampling(in_points: tf.Tensor,
                                      m: int) -> tf.Tensor:
    """

    Args:
        in_points: a `float32` Tensor with shape [b, n, 3]
        m: the number of output sampled points

    Returns:
        an `int32` Tensor with shape [b, m] to indicate the indices of m points
    """
    return graphics_tf_module.iterative_farthest_point_sample(in_points, m)


tf.no_gradient('iterative_farthest_point_sample')


def query_ball_point(in_points: tf.Tensor,
                     query_points: tf.Tensor,
                     radius: float,
                     max_points: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """

    Args:
        in_points: a `float` Tensor with shape [b, n, 3] to save the location of each in-point
        query_points: a `float` Tensor with shape [b, m, 3] to save the location of each query point
        radius: the valid radius to decide whether a point is fallen into a query point
        max_points: the maximum saving number of indices from in-points for each query point

    Returns:
        an `int32` Tensor with shape [b, m, max_points] to indicate the indices in in_points
        an `int32` Tensor with shape [b, m] to save the total number of points fallen into each query point
    """
    return graphics_tf_module.query_ball_point(in_points, query_points, radius, max_points)


class TestSampling(tf.test.TestCase):
    def __init__(self, method_name='TestSampling'):
        super().__init__(methodName=method_name)
        self.resource_dir = f'{os.path.split(__file__)[0]}/../unittest_resource'

    def testImportanceCoverageSamplingShape(self):
        n = 8
        b = 4
        t_shape = [6, 6]
        label_ref_ph = tf.compat.v1.placeholder(tf.float32, name='label_ref_ph', shape=[b, *t_shape, 20])
        logit_ref_ph = tf.compat.v1.placeholder(tf.float32, name='logit_ref_ph', shape=[b, *t_shape, 20])
        sample_op = importance_coverage_sampling(label_ref_ph, logit_ref_ph, n)
        self.assertAllEqual(sample_op.shape.as_list(), [b, n, len(t_shape)])

    @staticmethod
    def make_one_hot(data, channel_num=21):
        return np.eye(channel_num, dtype=np.int32)[data]

    def testImportanceCoverageSampling(self):
        b, n = 2, 128
        k, beta, seed = 100, 0.75, 0
        # k, beta, seed = 100, 1, 0

        sampled_pts_gt_name = f'importance_coverage_sampling_output_n-{n}_k-{k}_beta-{beta}_seed-{seed}.npz'
        sampled_pts_gt = np.load(os.path.join(self.resource_dir, sampled_pts_gt_name))['data']

        vol_data_list = np.load(os.path.join(self.resource_dir, 'unittest_input.npz'))['data']
        vol_label = self.make_one_hot(vol_data_list)
        vol_label = tf.compat.v1.convert_to_tensor(vol_label, tf.float32)
        vol_label_pad = tf.pad(vol_label, tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]), 'REFLECT')
        vol_logit = stylegan.PGGANOps.blur(vol_label_pad, padding='VALID')
        sample_shape = np.array(vol_label.shape[1:-1].as_list()) - 1
        sample_op = importance_coverage_sampling(vol_label, vol_logit, n, sample_shape, k=k, beta=beta, seed=seed)

        with tf.compat.v1.Session() as sess:
            sampled_pts = sess.run(sample_op)
            self.assertAllClose(sampled_pts, sampled_pts_gt)

    def load_test_data_list(self, sampled_gt_name, n):
        sampled_gt = np.load(os.path.join(self.resource_dir, sampled_gt_name))['data']

        vol_data_list = np.load(os.path.join(self.resource_dir, 'unittest_input.npz'))['data']
        vol_points_list = np.stack([np.argwhere(vol_data > 0)[:n] for vol_data in vol_data_list], axis=0)
        in_points = tf.convert_to_tensor(vol_points_list, tf.float32)
        return sampled_gt, in_points

    def testIterativeFarthestPointSample(self):
        n, m = 1024, 64
        # n, m = 1024, 128

        sampled_gt_name = f'iterative_farthest_point_sample_output_n-{n}_m-{m}.npz'
        sampled_gt, in_points = self.load_test_data_list(sampled_gt_name, n)
        sample_points_indices_ops = iterative_farthest_point_sampling(in_points, m)

        with self.test_session():
            sampled_pts_indices = sample_points_indices_ops.eval()
            self.assertAllClose(sampled_pts_indices, sampled_gt)

    def testQueryBallPoint(self):
        n, m = 1024, 16
        radius, max_points = 3, 512
        # radius, max_points = 4, 32

        query_gt_name = f'query_ball_point_output_n-{n}_m-{m}_radius-{radius}_max_points-{max_points}.npz'
        query_gt, in_points = self.load_test_data_list(query_gt_name, n)
        sample_pts_indices = iterative_farthest_point_sampling(in_points, m)

        batch_index = np.arange(in_points.shape.dims[0].value)[:, np.newaxis]
        batch_index = tf.convert_to_tensor(np.tile(batch_index, [1, m]), tf.int32)
        batch_pts_indices = tf.stack((batch_index, sample_pts_indices), axis=-1)
        sampled_pts = tf.gather_nd(in_points, batch_pts_indices)

        query_pts_indices, query_pts_num = query_ball_point(in_points, sampled_pts, radius, max_points)

        with self.test_session():
            query_pts_indices = query_pts_indices.eval()
            self.assertAllClose(query_pts_indices, query_gt)
