"""Formatoption widgets for psy-maps"""
import contextlib
from PyQt5 import QtWidgets, QtGui
import psy_simple.widgets.colors as psyps_wcol


class LSMFmtWidget(QtWidgets.QWidget):
    """The widget for the land-sea-mask formatoption"""

    def __init__(self, parent, fmto, project):
        super().__init__()
        import cartopy.feature as cf
        self.editor = parent

        self.cb_land = QtWidgets.QCheckBox('Land')
        self.cb_ocean = QtWidgets.QCheckBox('Ocean')
        self.cb_coast = QtWidgets.QCheckBox('Coastlines')

        self.land_color = psyps_wcol.ColorLabel(cf.LAND._kwargs['facecolor'])
        self.ocean_color = psyps_wcol.ColorLabel(cf.OCEAN._kwargs['facecolor'])
        self.coast_color = psyps_wcol.ColorLabel('k')

        self.txt_linewidth = QtWidgets.QLineEdit()
        self.txt_linewidth.setValidator(QtGui.QDoubleValidator(0, 100, 4))
        self.txt_linewidth.setPlaceholderText('Linewidth of coastlines')
        self.txt_linewidth.setToolTip('Linewidth of coastlines')

        self.combo_resolution = QtWidgets.QComboBox()
        self.combo_resolution.addItems(['110m', '50m', '10m'])

        self.refresh(fmto.value)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.cb_land)
        hbox.addWidget(self.land_color)
        hbox.addWidget(self.cb_ocean)
        hbox.addWidget(self.ocean_color)
        hbox.addWidget(self.cb_coast)
        hbox.addWidget(self.coast_color)
        hbox.addWidget(self.txt_linewidth)
        hbox.addWidget(self.combo_resolution)
        self.setLayout(hbox)

        for cb in [self.cb_land, self.cb_ocean, self.cb_coast]:
            cb.stateChanged.connect(self.toggle_and_update)
        self.txt_linewidth.textEdited.connect(self.toggle_and_update)
        self.combo_resolution.currentIndexChanged.connect(
            self.toggle_and_update)
        for lbl in [self.land_color, self.ocean_color, self.coast_color]:
            lbl.color_changed.connect(self.toggle_and_update)

    @property
    def value(self):
        ret = {}
        if self.cb_land.isChecked():
            ret['land'] = list(self.land_color.color.getRgbF())
        if self.cb_ocean.isChecked():
            ret['ocean'] = list(self.ocean_color.color.getRgbF())
        if self.cb_coast.isChecked():
            ret['coast'] = list(self.coast_color.color.getRgbF())
            ret['linewidth'] = float(self.txt_linewidth.text().strip() or 0.0)
        if ret:
            ret['res'] = self.combo_resolution.currentText()
        return ret

    def toggle_and_update(self):
        self.toggle_color_labels()
        value = self.value
        self.editor.set_obj(value)

    def toggle_color_labels(self):
        self.land_color.setEnabled(self.cb_land.isChecked())
        self.ocean_color.setEnabled(self.cb_ocean.isChecked())

        self.coast_color.setEnabled(self.cb_coast.isChecked())
        self.txt_linewidth.setEnabled(self.cb_coast.isChecked())

    @contextlib.contextmanager
    def block_widgets(self, *widgets):
        widgets = widgets or [self.cb_land, self.cb_ocean, self.cb_coast,
                              self.land_color, self.ocean_color,
                              self.coast_color,
                              self.txt_linewidth, self.combo_resolution]
        for w in widgets:
            w.blockSignals(True)
        yield
        for w in widgets:
            w.blockSignals(False)

    def refresh(self, value):
        with self.block_widgets():
            self.cb_land.setChecked('land' in value)
            self.cb_ocean.setChecked('ocean' in value)
            self.cb_coast.setChecked('coast' in value)

            if 'linewidth' in value:
                self.txt_linewidth.setText(str(value['linewidth']))
            elif 'coast' in value:
                self.txt_linewidth.setText('1.0')
            else:
                self.txt_linewidth.setText('')

            if 'res' in value:
                self.combo_resolution.setCurrentText(value['res'])
            else:
                self.combo_resolution.setCurrentText('110m')

            if 'land' in value:
                self.land_color._set_color(value['land'])
            if 'ocean' in value:
                self.ocean_color._set_color(value['ocean'])
            if 'coast' in value:
                self.coast_color._set_color(value['coast'])

            self.toggle_color_labels()


class GridFmtWidget(psyps_wcol.CTicksFmtWidget):
    """The formatoption widget for xgrid and ygrid"""

    methods = ['Discrete', 'Auto', 'Disable']

    methods_type = psyps_wcol.BoundsType

    auto_val = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, properties=False)

    def set_value(self, value):
        if value is False or value is None:
            with self.block_widgets(self.method_combo, self.type_combo):
                self.type_combo.setCurrentText('Disable')
            self.refresh_methods('Disable')
        else:
            super().set_value(value)

    def refresh_methods(self, text):
        if text == 'Disable':
            with self.block_widgets(self.method_combo):
                self.method_combo.clear()
            self.set_obj(False)
            self.refresh_current_widget()
        else:
            super().refresh_methods(text)

    def refresh_current_widget(self):
        w = self.current_widget
        no_lines = self.type_combo.currentText() == 'Disable'
        if no_lines and w is not None:
            w.setVisible(False)
            self.current_widget = None
        if not no_lines:
            super().refresh_current_widget()