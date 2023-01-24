from setuptools import setup, find_packages

setup(
    name='image_processing_pipeline',
    version='',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    url='',
    license='',
    author='malchul',
    author_email='',
    description='',
    install_requires=['numpy==1.24.0',
                     ' tqdm==4.19.9',
                     ' omegaconf',
                     ' opencv-python==4.5.5.64',
                     ' opencv-contrib-python==4.5.5.64',
                      # -------- infer ------------- #
                     'mediapipe',
                     'onnxruntime',
                     ],
    python_requires='>=3.6',
)
