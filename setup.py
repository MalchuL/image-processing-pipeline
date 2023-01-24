from setuptools import setup

setup(
    name='image-processing-pipeline',
    version='',
    packages=['utils', 'pipeline',
              'pipeline.utils', 'pipeline.readers', 'pipeline.detectors', 'pipeline.detectors.lib',
              'pipeline.preprocess', 'pipeline.postprocess', 'pipeline.postprocess.merging_crops',
              'pipeline.stylization', 'pipeline.stylization.inference_engine',
              'pipeline.stylization.inference_engine.models'],
    url='',
    license='',
    author='malchul',
    author_email='',
    description=''
)
