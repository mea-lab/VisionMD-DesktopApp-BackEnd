# app/apps.py

from django.apps import AppConfig
from django.conf import settings
import os
os.environ['CUDA_DEVICE_ORDER']         = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']      = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_CPP_LOG_LEVEL'] = 'ERROR'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf, tensorflow_hub as hub
from tensorflow.keras import mixed_precision
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
tf.keras.backend.clear_session()
tf.keras.mixed_precision.set_global_policy('mixed_float16')


class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = "app"
    print("Tensorflow built with CUDA: ", tf.test.is_built_with_cuda())
    print("GPUs available:", tf.config.list_physical_devices('GPU'))

    def ready(self):
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)