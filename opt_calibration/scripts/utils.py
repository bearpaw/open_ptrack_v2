import numpy as np
from mayavi import mlab

def rot_x(t):
    '''
    Compute rotation matrix along x axix
    Args: t - radian of rotation angle
    '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

def rot_y(t):
    '''
    Compute rotation matrix along y axix
    Args: t - radian of rotation angle
    '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])
def rot_z(t):
    '''
    Compute rotation matrix along z axix
    Args: t - radian of rotation angle
    '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])
                     
class Plotter3D(object):
    def __init__(self, size=(400, 300)):
        self.size = size
        self.fig = None

    def _create_figure(self):
        if self.fig == None:
            self.fig = mlab.figure(size=self.size)

    def _color_bar(self, title):
        cb = mlab.colorbar(title=title, orientation='vertical',
            nb_labels=10, label_fmt='%.1f')
        cb.label_text_property.bold = False
        cb.label_text_property.italic = False
        cb.scalar_bar.unconstrained_font_size = True
        cb.label_text_property.font_size = 16

    def draw_axis(self, points, line_width=1.2):
        min_val, max_val = points.min(0), points.max(0)
        mlab.plot3d([0, max_val[0]], [0, 0], [0, 0], color=(1, 0, 0), tube_radius=None, line_width=line_width, figure=self.fig)
        mlab.plot3d([0, 0], [0, max_val[1]], [0, 0], color=(0, 1, 0), tube_radius=None, line_width=line_width, figure=self.fig)
        mlab.plot3d([0, 0], [0, 0], [0, max_val[2]], color=(0, 0, 1), tube_radius=None, line_width=line_width, figure=self.fig)

    def draw_points(self, points, color=(1, 1, 1), label=None, prob=None,
        mode='point', scale_factor=1.2, title='', rgb=None):
        self._create_figure()
        # if points.ndim == 3:
        #     points = points.copy().squeeze()
        (x, y, z) = (points[:, 0], points[:, 1], points[:, 2])


        if isinstance(color, np.ndarray):
            color = tuple(color.tolist())

        if isinstance(color, list):
            color = tuple(color)

        if label is not None:
            label_val = np.unique(label).tolist()
            colors = sns.color_palette('coolwarm', len(label_val))
            for l in label_val:
                idx = (label == l).nonzero()
                color = tuple([c for c in colors[l]])
                mlab.points3d(x[idx], y[idx], z[idx], color=color, scale_factor=scale_factor)
        elif prob is not None:
            mlab.points3d(x, y, z, prob, colormap='jet', scale_factor=scale_factor, scale_mode='vector')
            self._color_bar(title)
        else:
            if rgb is None:
                mlab.points3d(x, y, z, color=color, mode=mode, scale_factor=scale_factor)
            else:
                rgba = np.concatenate((rgb, np.ones((rgb.shape[0], 1))*255), axis=1).astype(np.uint8)
                pts = mlab.pipeline.scalar_scatter(x, y, z, mode='point') # plot the points
                pts.add_attribute(rgba, 'colors') # assign the colors to each point
                pts.data.point_data.set_active_scalars('colors')
                g = mlab.pipeline.glyph(pts)
                g.glyph.glyph.scale_factor = scale_factor # set scaling for all the points
                g.glyph.scale_mode = 'data_scaling_off' # make all the points same size

        mlab.orientation_axes()


    def draw_boxes(self, boxes, color=(1,1,1), line_width=1, draw_text=True, text_scale=(1,1,1), color_list=None):
        ''' Draw 3D bounding boxes
        Args:
            boxes: numpy array (n,8,3) for XYZs of the box corners
            fig: mayavi figure handler
            color: RGB value tuple in range (0,1), box line color
            line_width: box line width
            draw_text: boolean, if true, write box indices beside boxes
            text_scale: three number tuple
            color_list: a list of RGB tuple, if not None, overwrite color.
        Returns:
            fig: updated fig
        '''
        self._create_figure()
        num = len(boxes)
        for n in range(num):
            b = boxes[n]
            if color_list is not None:
                color = color_list[n]
            if draw_text: mlab.text3d(b[4,0], b[4,1], b[4,2], '%d'%n, scale=text_scale, color=color, figure=fig)
            for k in range(0,4):
                #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
                i,j=k,(k+1)%4
                mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

                i,j=k+4,(k+1)%4 + 4
                mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

                i,j=k,k+4
                mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

    def draw_poses(self, poses, skeletons=None, colors=None, tube_radius=2):
        '''
        '''
        self._create_figure()
        for pose in poses:
            if pose is not None and skeletons is not None:
                for id, sk in enumerate(skeletons):
                    px, py, pz = pose[sk, 0], pose[sk, 1], pose[sk, 2]
                    c = tuple([c for c in colors[id]])
                    mlab.plot3d(px, py, pz, color=c, tube_radius=tube_radius)

    def plot(self, ply, pose=None, skeletons=None, skeleton_colors=None, num_point=1024,
        tube_radius=2, is_show=False, is_draw=False, file_path='temp.png',
        label = None, num_joints=19):
        '''
        Plot vertices and triangles from a PlyData instance. Assumptions:
            `ply' has a 'vertex' element with 'x', 'y', and 'z'
                properties;
            `ply' has a 'face' element with an integral list property
                'vertex_indices', all of whose elements have length 3.
        '''

        self._create_figure()
        mlab.clf()

        if isinstance(ply, PlyData):
            vertex = ply['vertex']
            (x, y, z) = (vertex[t][0:num_point] for t in ('x', 'y', 'z'))
        elif isinstance(ply, np.ndarray):
            ply = ply.copy().squeeze()[0:num_point, :]
            (x, y, z) = (ply[:, 0], ply[:, 1], ply[:, 2])
        else:
            print('Only numpy.array and PlyData are supported: ', type(ply))
            return

        if label is not None:
            assert label.size == num_point
            label_val = np.unique(label).tolist()
            colors = sns.color_palette('coolwarm', num_joints)
            for l in label_val:
                idx = (label == l).nonzero()
                color = tuple([c for c in colors[l]])
                mlab.points3d(x[idx], y[idx], z[idx], color=color, scale_factor=1.2)
        else:
            mlab.points3d(x, y, z, color=(1, 1, 1), mode='point')

        # plot origin=
        mlab.points3d(0, 0, 0, 0.5, color=(1, 1, 1), scale_factor=10, transparent=True)

        if pose is not None and skeletons is not None:
            for id, sk in enumerate(skeletons):
                px, py, pz = pose[sk, 0], pose[sk, 1], pose[sk, 2]
                c = tuple([c for c in skeleton_colors[id]])
                mlab.plot3d(px, py, pz, color=c, tube_radius=tube_radius)

        mlab.view(azimuth=180, elevation=0, distance=800, focalpoint=[8, -100, 8])

        if is_show:
            mlab.show(stop=True)

        if is_draw:
            mlab.savefig(file_path)

    def show(self, stop=True):
        mlab.show(stop=stop)

    def screenshot(self):
        return mlab.screenshot()

    def save(self, file_path):
        # mkdir_p(os.path.dirname(file_path))
        mlab.savefig(file_path)

    def view(self, azimuth=None, elevation=None, distance=None, focalpoint=None, roll=None, reset_roll=True, figure=None):
        return mlab.view(azimuth=azimuth, elevation=elevation, distance=distance,
            focalpoint=focalpoint, roll=roll, reset_roll=reset_roll, figure=figure)

    def move(self, forward=None, right=None, up=None):
        return mlab.move(forward=forward, right=right, up=up)

    def clf(self):
        if self.fig is not None:
            mlab.clf()
