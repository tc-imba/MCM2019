import pylab as pl
import numpy as np

from mayavi import mlab
from tvtk.api import tvtk

# state = 'OH'
# substance_name = 'Buprenorphine'
state = 'KY'
substance_name = 'Hydrocodone'


def draw_layer(path, position):
    # load a png with a scale 0->1 and four color channels (an extra alpha channel for transparency).
    im = pl.imread(path, format='png') * 255

    colors = tvtk.UnsignedCharArray()
    colors.from_array(im.transpose((1, 0, 2)).reshape(-1, 4))

    m_image = mlab.imshow(
        np.ones(im.shape[:2]),
        extent=[0, 0, 0, 0, position, position],
        opacity=0.6)

    m_image.actor.input.point_data.scalars = colors


if __name__ == "__main__":
    fig_num = 5
    fig_height = 200 * (fig_num - 1)

    mlab.figure(bgcolor=(0.97, 0.97, 0.97), size=(1000, 1000))

    for i in range(fig_num):
        draw_layer('../figure/%s_%s_%d.png' % (state, substance_name, i + 1), i * 200)

    im = pl.imread('../figure/%s_%s_1.png' % (state, substance_name), format='png')

    m_image = mlab.quiver3d(-im.shape[0] / 2, -im.shape[1] / 2, fig_height,
                            0, 0, -fig_height - 50,
                            line_width=.1, colormap='Blues',
                            scale_factor=1, mode='arrow', resolution=25)

    m_image.glyph.glyph_source.glyph_source.shaft_radius = 0.005
    m_image.glyph.glyph_source.glyph_source.tip_length = 0.02
    m_image.glyph.glyph_source.glyph_source.tip_radius = 0.01

    mlab.view(azimuth=30, elevation=62)
    mlab.gcf().scene.parallel_projection = True
    mlab.gcf().scene.camera.zoom(1.1)

    mlab.draw()
    mlab.savefig('../figure/%s_%s.png' % (state, substance_name))
    mlab.show()
