"""Test module for formatoption widgets"""
import os.path as osp
import pytest
from PyQt5.QtCore import Qt


@pytest.fixture
def test_ds():
    import psyplot.data as psyd
    test_file = osp.join(osp.dirname(__file__), '..', 'test-t2m-u-v.nc')
    with psyd.open_dataset(test_file) as ds:
        yield ds


@pytest.fixture
def sp(test_ds):
    with test_ds.psy.plot.mapplot(name='t2m') as sp:
        yield sp


@pytest.fixture
def plotter(sp):
    return sp.plotters[0]


@pytest.fixture
def mainwindow(qtbot):
    from psyplot_gui.main import MainWindow, rcParams
    with rcParams.catch():
        rcParams['console.start_channels'] = False
        rcParams['main.listen_to_port'] = False
        rcParams['help_explorer.render_docs_parallel'] = False
        rcParams['help_explorer.use_intersphinx'] = False
        window = MainWindow(show=False)
        qtbot.addWidget(window)
        yield window


def test_lsm_fmt_widget(mainwindow, plotter, qtbot):
    from psy_maps.widgets import LSMFmtWidget
    mainwindow.fmt_widget.fmto = plotter.lsm

    w = mainwindow.fmt_widget.fmt_widget

    assert isinstance(w, LSMFmtWidget)

    assert not 'land' in mainwindow.fmt_widget.get_obj()
    assert not w.cb_land.isChecked()

    w.cb_land.setChecked(True)

    assert w.cb_land.isChecked()
    assert 'land' in mainwindow.fmt_widget.get_obj()

    w.combo_resolution.setCurrentText('10m')

    assert mainwindow.fmt_widget.get_obj()['res'] == '10m'

    w.cb_coast.setChecked(False)

    assert 'linewidth' not in mainwindow.fmt_widget.get_obj()
