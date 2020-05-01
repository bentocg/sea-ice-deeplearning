import sys
from argparse import ArgumentParser

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from utils.training_set_extraction import PatchNavigator


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--scn_dir', type=str)
    arg_parser.add_argument('--patch_size', type=int)
    arg_parser.add_argument('--out_dir', type=str)
    return arg_parser.parse_args()


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'PyQt5 matplotlib example - pythonspot.com'
        self.width = 1980
        self.height = 1080
        self.patch = PlotCanvas(self, width=10, height=8)
        self.mask = PlotCanvas(self, width=10, height=8)
        self.scene = PlotCanvas(self, width=10, height=8)
        self.init_ui()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_K:
            patch_navigator.write_patch('keep')
            self.plot_patches()

        elif event.key() == QtCore.Qt.Key_J:
            patch_navigator.write_patch('negative')
            self.plot_patches()

        elif event.key() == QtCore.Qt.Key_L:
            patch_navigator.write_patch('positive')
            self.plot_patches()

        elif event.key() == QtCore.Qt.Key_Space:
            patch_navigator.skip_cells()
            self.plot_patches()

        elif event.key() == QtCore.Qt.Key_Escape:
            self.close()

        elif event.key() == QtCore.Qt.Key_S:
            patch_navigator.next_scene()
            self.plot_patches()

        elif event.key() == QtCore.Qt.Key_A:
            patch_navigator.previous_scene()
            self.plot_patches()

        elif event.key() == QtCore.Qt.Key_R:
            patch_navigator.next_row()
            self.plot_patches()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.plot_patches()
        self.show()

    def plot_patches(self):

        self.patch.plot_patch()
        self.patch.move(0, 450)

        self.mask.plot_mask()
        self.mask.move(600, 450)

        self.scene.plot_scene()
        self.scene.move(0, -70)


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=70):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot_patch(self):
        self.axes.cla()
        self.axes.imshow(patch_navigator.curr_patch, cmap='gray', vmin=0, vmax=255)
        self.draw()

    def plot_mask(self):
        self.axes.cla()
        self.axes.imshow(patch_navigator.curr_mask, cmap='gray', vmin=0, vmax=255)
        self.draw()

    def plot_scene(self):
        self.axes.cla()
        self.axes.imshow(patch_navigator.curr_scn, cmap='gray', vmin=0, vmax=255)
        self.axes.plot([patch_navigator.j * patch_navigator.patch_size / patch_navigator.scn_ratio[0]],
                       [patch_navigator.i * patch_navigator.patch_size / patch_navigator.scn_ratio[1]], 'rs',
                       markersize=18, linewidth=2, alpha=0.5)
        self.draw()


if __name__ == '__main__':
    args = parse_args()
    patch_navigator = PatchNavigator(scenes_dir=args.scn_dir,
                                     patch_size=args.patch_size,
                                     out_dir=args.out_dir,
                                     scn_res=(1000, 1000))

    app = QApplication(sys.argv)
    ex = App()

    sys.exit(app.exec_())
