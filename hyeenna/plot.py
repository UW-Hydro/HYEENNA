from IPython.core.display import display, HTML
from string import Template
import json
import os
import numpy as np

FILE_PATH = __file__
JS_PATH = os.path.join(os.path.dirname(FILE_PATH), 'js')
with open(os.path.join(JS_PATH, 'chord_plot.js'), 'r') as f:
    JS_CODE = f.read()

BASE_HTML = '''
    <script>
      require.config({paths: {d3: "https://d3js.org/d3.v4.min"}});
      require(["d3"], function(d3) {
          $JS_CODE
          chord_plot($DATA, $NAMES, $COLORS);
      });
    </script>
    <div id=diagram></div>
    '''


class NumpyEncoder(json.JSONEncoder):
    """
    Credit:
    https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def plot_chords(data, names, colors):
    return HTML(Template(BASE_HTML).substitute({
            'JS_CODE': JS_CODE,
            'DATA': json.dumps(data, cls=NumpyEncoder),
            'NAMES': json.dumps(names, cls=NumpyEncoder),
            'COLORS': json.dumps(colors, cls=NumpyEncoder)
        }))
