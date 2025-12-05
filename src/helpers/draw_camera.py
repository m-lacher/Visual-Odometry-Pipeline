import numpy as np

from src.helpers.arrow_3d import Arrow3D

def drawCamera(ax, position, direction, length_scale=1, head_size=10, 
               equal_axis=True, set_ax_limits=True):
    position = np.asarray(position).reshape(3,)
    direction = np.asarray(direction).reshape(3,3)

    arrow_prop_dict = dict(mutation_scale=head_size, arrowstyle='-|>', color='r')
    a = Arrow3D([position[0], position[0] + length_scale * direction[0, 0]],
                [position[1], position[1] + length_scale * direction[1, 0]],
                [position[2], position[2] + length_scale * direction[2, 0]],
                **arrow_prop_dict)
    ax.add_artist(a)

    arrow_prop_dict['color'] = 'g'
    a = Arrow3D([position[0], position[0] + length_scale * direction[0, 1]],
                [position[1], position[1] + length_scale * direction[1, 1]],
                [position[2], position[2] + length_scale * direction[2, 1]],
                **arrow_prop_dict)
    ax.add_artist(a)

    arrow_prop_dict['color'] = 'b'
    a = Arrow3D([position[0], position[0] + length_scale * direction[0, 2]],
                [position[1], position[1] + length_scale * direction[1, 2]],
                [position[2], position[2] + length_scale * direction[2, 2]],
                **arrow_prop_dict)
    ax.add_artist(a)

    if not set_ax_limits:
        return

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    ax.set_xlim([min(xlim[0], position[0]), max(xlim[1], position[0])])
    ax.set_ylim([min(ylim[0], position[1]), max(ylim[1], position[1])])
    ax.set_zlim([min(zlim[0], position[2]), max(zlim[1], position[2])])

    if equal_axis:
        ax.set_box_aspect((np.ptp(ax.get_xlim()),
                           np.ptp(ax.get_ylim()),
                           np.ptp(ax.get_zlim())))

