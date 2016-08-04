# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import pandas as pd
import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep",
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 400, "Number of training steps.")
flags.DEFINE_string(
    "train_data",
    "",
    "Path to the training data.")
flags.DEFINE_string(
    "test_data",
    "",
    "Path to the test data.")

COLUMNS = ["Semana", "Agencia_ID", "Canal_ID", "Ruta_SAK", "Cliente_ID",
           "Producto_ID", "Demanda_uni_equil"]
LABEL_COLUMN = "Demanda_uni_equil"
CATEGORICAL_COLUMNS = ["Agencia_ID", "Canal_ID", "Ruta_SAK", "Cliente_ID", "Producto_ID"]
CONTINUOUS_COLUMNS = ["Semana"]

def build_estimator(model_dir):

    agencia_id = tf.contrib.layers.sparse_column_with_hash_bucket("Agencia_ID", hash_bucket_size=10)
    cancal_id = tf.contrib.layers.sparse_column_with_hash_bucket("Canal_ID", hash_bucket_size=10)
    ruta_sak = tf.contrib.layers.sparse_column_with_hash_bucket("Ruta_SAK", hash_bucket_size=10)
    cliente_id = tf.contrib.layers.sparse_column_with_hash_bucket("Cliente_ID", hash_bucket_size=4000)
    producto_id = tf.contrib.layers.sparse_column_with_hash_bucket("Producto_ID", hash_bucket_size=1000)

    # Continuous base columns.
    semana = tf.contrib.layers.real_valued_column("Semana")
    # Wide columns and deep columns.
    deep_columns = [
      tf.contrib.layers.embedding_column(agencia_id, dimension=4),
      tf.contrib.layers.embedding_column(cancal_id, dimension=4),
      tf.contrib.layers.embedding_column(ruta_sak, dimension=4),
      tf.contrib.layers.embedding_column(cliente_id, dimension=32),
      tf.contrib.layers.embedding_column(producto_id, dimension=16),
      semana,
    ]

    m = tf.contrib.learn.DNNRegressor(model_dir=model_dir, feature_columns=deep_columns,hidden_units=[200, 100], optimizer=tf.train.AdamOptimizer(0.1))
    return m


def input_fn(df):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label


def train_and_eval():
    """Train and evaluate the model."""

    df_train = pd.read_csv('bimbo_train.csv', delimiter=',',
                           dtype={'Semana':np.int32, 'Agencia_ID': str, 'Canal_ID': str,
                                  'Ruta_SAK': str, 'Cliente_ID': object, 'Producto_ID': str, 'Demanda_uni_equil': np.int32})

    df_eval = pd.read_csv('bimbo_val.csv', delimiter=',',
                          dtype={'Semana':np.int32, 'Agencia_ID': str, 'Canal_ID': str,
                                 'Ruta_SAK': str, 'Cliente_ID': object, 'Producto_ID': str, 'Demanda_uni_equil': np.int32})

    model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
    print("model directory = %s" % model_dir)

    m = build_estimator(model_dir)
    m.fit(input_fn=lambda: input_fn(df_train), steps=FLAGS.train_steps)
    preds = m.predict(input_fn=lambda: input_fn(df_eval))
    rmlse = np.sqrt(np.mean(np.square(np.log1p(preds) - np.log1p(df_eval['Demanda_uni_equil']))))
    # print (preds)
    print (rmlse)
    # results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    # for key in sorted(results):
    #    print("%s: %s" % (key, results[key]))


def main(_):
    train_and_eval()


if __name__ == "__main__":
    tf.app.run()
