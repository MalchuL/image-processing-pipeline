class ImagePipelineException(Exception):
    ...


class InstantiationException(ImagePipelineException):
    ...


class PipelineConfigurationError(ImagePipelineException):
    ...