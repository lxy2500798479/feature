"""
核心异常模块
"""

class BaseAPIException(Exception):
    """基础 API 异常"""

    def __init__(self, message: str, code: int = 500):
        self.message = message
        self.code = code
        super().__init__(self.message)


class DocumentNotFoundException(BaseAPIException):
    """文档未找到异常"""

    def __init__(self, doc_id: str):
        super().__init__(f"文档未找到: {doc_id}", 404)


class ProcessingException(BaseAPIException):
    """处理异常"""

    def __init__(self, message: str):
        super().__init__(f"处理失败: {message}", 500)


class ValidationException(BaseAPIException):
    """验证异常"""

    def __init__(self, message: str):
        super().__init__(f"验证失败: {message}", 400)


class StorageException(BaseAPIException):
    """存储异常"""

    def __init__(self, message: str):
        super().__init__(f"存储失败: {message}", 500)
